// r4d_module.hip: the pybind surface for the R4D kernel library. Every kernel in the library is
// reachable from here and nowhere else -- the kernels themselves are plain HIP behind the C ABI in
// r4d.h, so the same sources also compile without Python.
//
// The Python names are the C names with the r4d_ prefix dropped, and nothing else. They carry the
// geometry each kernel is compiled for (see the naming note in r4d.h) so that binding one is a
// deliberate act: a caller picks `attn_decode_h256_gqa6_fp8kv` because its model IS head_dim 256
// with a 6:1 GQA ratio and an fp8 cache, and a model that is not will not silently pick it up.
//
//     attn_prefill_h256_gqa6_fp8kv   attn_prefill_h256_gqa6_bf16kv
//     attn_decode_h256_gqa6_fp8kv    attn_decode_h256_gqa6_bf16kv
//     attn_vit_h72_bf16
//     gdn_chunk_scan_k128_v128_c64_bf16, gdn_kkt_solve_k128_c64_bf16,
//     gdn_conv_prep_w4_h128_bf16, gdn_conv_update_w4_h128_bf16,
//     gdn_recurrent_update_k128_v128_bf16_fp32state
//     ar_oneshot_2rank_exact         ar_oneshot_2rank_wht6
//     gemm_bf16_nt_m16
//
// `kernels()` returns the same set with the constraints spelled out, and the module constants
// (ATTN_*, GDN_*, AR_*, GEMM_*) report the compiled-in geometry, so a caller can test a model
// against this build instead of hardcoding a second copy of the numbers. `select(op, **geometry)`
// does that test for the caller and returns the kernel name or None -- None meaning "use whatever
// you would have used without R4D", which is the answer for every model this build was not
// compiled for. `explain()` says which constraint failed, and `selections()` replays every question
// select() has been asked in this process -- which is how a serve reports the kernels it resolved
// to, long after the resolving happened at import time.
//
// Every entry point takes device addresses the caller already owns and the stream to launch on.
// Nothing here allocates, synchronises, or reads device memory from the host (the IPC helpers are
// startup-only), so a launch is safe to record into a HIP graph -- which is what happens on the
// decode path downstream, where the whole model step is captured.
#include "r4d.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace py = pybind11;

// =================================================================================================
// attention
// =================================================================================================
// Argument order is the same for all four kernels:
//   q, kv, block_table, seqused_k, out, k_descale, v_descale, scratch,
//   num_seqs, q_len, q_heads, kv_heads, head_dim, block_size, max_blocks,
//   kv_block_stride, kv_head_stride, scale, splits, max_ctx, stream
// `scratch`/`splits` are decode-only (prefill needs neither); `splits` 0 lets the kernel's own
// split law choose. Strides are in ELEMENTS of the cache dtype, not bytes.
static R4DArgs mk_args(int64_t q, int64_t kv, int64_t block_table, int64_t seqused_k, int64_t out,
                       int64_t k_descale, int64_t v_descale, int64_t scratch, int num_seqs,
                       int q_len, int q_heads, int kv_heads, int head_dim, int block_size,
                       int max_blocks, int64_t kv_block_stride, int64_t kv_head_stride,
                       double scale, int splits, int max_ctx) {
  R4DArgs a{};
  a.q               = (const void*)q;
  a.kv              = (const void*)kv;
  a.block_table     = (const int*)block_table;
  a.seqused_k       = (const int*)seqused_k;
  a.out             = (void*)out;
  a.k_descale       = (const float*)k_descale;   // null => 1.0, folded away
  a.v_descale       = (const float*)v_descale;
  a.q_descale       = nullptr;                   // the query is bf16 in every variant
  a.scratch         = (void*)scratch;
  a.num_seqs        = num_seqs;
  a.q_len           = q_len;
  a.q_heads         = q_heads;
  a.kv_heads        = kv_heads;
  a.head_dim        = head_dim;
  a.block_size      = block_size;
  a.max_blocks      = max_blocks;
  a.kv_block_stride = kv_block_stride;
  a.kv_head_stride  = kv_head_stride;
  a.scale           = (float)scale;
  a.splits          = splits;
  a.max_ctx         = max_ctx;
  return a;
}

// The C ABI reports unsupported shapes as negative codes rather than aborting; turn them into a
// Python exception so a bad launch surfaces at the call site instead of as wrong numbers. The
// message names the constraint the kernel's own name advertises.
static void check(int rc, const char* what) {
  if (rc == 0) return;
  const char* why = "unsupported shape";
  switch (rc) {
    case -1: why = "head_dim must be 256 and the paged block size 16"; break;
    case -2: why = "there must be exactly 6 queries per kv head"; break;
    case -4: why = "q_len * gqa exceeds the 64 query rows one decode workgroup holds"; break;
    default: if (rc <= -100) why = "hip launch error"; break;
  }
  throw std::runtime_error(std::string(what) + ": " + why + " (code " + std::to_string(rc) + ")");
}

// KIND: 0 = prefill, 1 = decode. KVP: 0 = fp8-e4m3 cache, 1 = bf16 cache. One instantiation per
// bound name, so the Python name and the kernel that runs are the same choice.
template <int KIND, int KVP>
static void attn(int64_t q, int64_t kv, int64_t block_table, int64_t seqused_k, int64_t out,
                 int64_t k_descale, int64_t v_descale, int64_t scratch, int num_seqs, int q_len,
                 int q_heads, int kv_heads, int head_dim, int block_size, int max_blocks,
                 int64_t kv_block_stride, int64_t kv_head_stride, double scale, int splits,
                 int max_ctx, int64_t stream) {
  R4DArgs a = mk_args(q, kv, block_table, seqused_k, out, k_descale, v_descale, scratch, num_seqs,
                      q_len, q_heads, kv_heads, head_dim, block_size, max_blocks, kv_block_stride,
                      kv_head_stride, scale, splits, max_ctx);
  const hipStream_t s = (hipStream_t)stream;
  int rc;
  if (KIND == 0) rc = KVP ? r4d_attn_prefill_h256_gqa6_bf16kv(&a, s)
                          : r4d_attn_prefill_h256_gqa6_fp8kv(&a, s);
  else           rc = KVP ? r4d_attn_decode_h256_gqa6_bf16kv(&a, s)
                          : r4d_attn_decode_h256_gqa6_fp8kv(&a, s);
  check(rc, KIND == 0 ? (KVP ? "r4d attn_prefill_h256_gqa6_bf16kv"
                             : "r4d attn_prefill_h256_gqa6_fp8kv")
                      : (KVP ? "r4d attn_decode_h256_gqa6_bf16kv"
                             : "r4d attn_decode_h256_gqa6_fp8kv"));
}

// Bytes of split-KV partial buffer one decode launch of this shape needs. The caller sizes its
// persistent scratch from the worst case ONCE, at startup: allocating inside a graph capture is
// not allowed, and the buffer has to be the same address at capture and at replay.
static int64_t attn_decode_scratch_bytes(int num_seqs, int q_len, int q_heads, int kv_heads,
                                         int head_dim, int max_ctx, int splits) {
  R4DArgs a = mk_args(0, 0, 0, 0, 0, 0, 0, 0, num_seqs, q_len, q_heads, kv_heads, head_dim, 16, 0,
                      0, 0, 1.0, splits, max_ctx);
  return r4d_attn_decode_h256_gqa6_scratch_bytes(&a);
}

// =================================================================================================
// gated delta net
// =================================================================================================
static void gdn_chunk_scan(int64_t q, int64_t k, int64_t v, int64_t A, int64_t g, int64_t beta,
                           int64_t h0, int64_t o, int64_t ht, int64_t cu, int num_seqs,
                           int num_v_heads, int num_k_heads, int head_k, int head_v, int chunk,
                           double scale, int64_t stream) {
  int rc = r4d_gdn_chunk_scan_k128_v128_c64_bf16(
      (const void*)q, (const void*)k, (const void*)v, (const void*)A, (const void*)g,
      (const void*)beta, (const void*)h0, (void*)o, (void*)ht, (const void*)cu, num_seqs,
      num_v_heads, num_k_heads, head_k, head_v, chunk, (float)scale, (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_chunk_scan_k128_v128_c64_bf16: this build is compiled for "
                             "head_k/head_v/chunk 128/128/64");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_chunk_scan_k128_v128_c64_bf16: hip launch error (code " +
                             std::to_string(rc) + ")");
}

static void gdn_kkt_solve(int64_t k, int64_t beta, int64_t g, int64_t A, int64_t cu, int num_seqs,
                          int num_tokens, int num_v_heads, int num_k_heads, int head_k, int chunk,
                          int64_t stream) {
  int rc = r4d_gdn_kkt_solve_k128_c64_bf16((const void*)k, (const void*)beta, (const void*)g,
                                           (void*)A, (const void*)cu, num_seqs, num_tokens,
                                           num_v_heads, num_k_heads, head_k, chunk,
                                           (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_kkt_solve_k128_c64_bf16: this build is compiled for "
                             "head_k 128 and chunk 64");
  if (rc == -2 || rc == -3)
    throw std::runtime_error("r4d gdn_kkt_solve_k128_c64_bf16: v heads must be a supported "
                             "multiple of k heads");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_kkt_solve_k128_c64_bf16: hip launch error (code " +
                             std::to_string(rc) + ")");
}

static void gdn_conv_prep(int64_t x, int64_t xpitch, int64_t wgt, int64_t bias, int64_t cstate,
                          int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, int64_t cache_idx,
                          int64_t ci_stride, int64_t has_init, int64_t a, int64_t b,
                          int64_t ab_stride, bool ab_is_bf16, int64_t A_log, int64_t dt_bias,
                          int64_t q, int64_t k, int64_t v, int64_t g, int64_t beta, int64_t cu,
                          int num_seqs, int num_tokens, int num_v_heads, int num_k_heads,
                          int head_k, int head_v, int width, double softplus_thr,
                          int64_t stream) {
  int rc = r4d_gdn_conv_prep_w4_h128_bf16(
      (const void*)x, xpitch, (const void*)wgt, (const void*)bias, (void*)cstate, cs_seq, cs_dim,
      cs_tok, (const void*)cache_idx, ci_stride, (const void*)has_init, (const void*)a,
      (const void*)b, ab_stride, ab_is_bf16 ? 1 : 0, (const void*)A_log, (const void*)dt_bias,
      (void*)q, (void*)k, (void*)v, (void*)g, (void*)beta, (const void*)cu, num_seqs, num_tokens,
      num_v_heads, num_k_heads, head_k, head_v, width, (float)softplus_thr, (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_conv_prep_w4_h128_bf16: this build is compiled for "
                             "head dim 128 and conv width 4");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_conv_prep_w4_h128_bf16: hip launch error (code " +
                             std::to_string(rc) + ")");
}

static void gdn_conv_update(int64_t x, int64_t xpitch, int64_t wgt, int64_t bias, int64_t cstate,
                            int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, int state_len_max,
                            int64_t cache_idx, int64_t ci_stride, int64_t num_accepted,
                            int64_t q, int64_t k, int64_t v, int64_t cu, int num_seqs,
                            int num_v_heads, int num_k_heads, int head_k, int head_v, int width,
                            int max_query_len, int64_t stream) {
  int rc = r4d_gdn_conv_update_w4_h128_bf16(
      (const void*)x, xpitch, (const void*)wgt, (const void*)bias, (void*)cstate, cs_seq, cs_dim,
      cs_tok, state_len_max, (const void*)cache_idx, ci_stride, (const void*)num_accepted,
      (void*)q, (void*)k, (void*)v, (const void*)cu, num_seqs, num_v_heads, num_k_heads, head_k,
      head_v, width, max_query_len, (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_conv_update_w4_h128_bf16: this build is compiled for "
                             "head dim 128 and conv width 4");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_conv_update_w4_h128_bf16: hip launch error (code " +
                             std::to_string(rc) + ")");
}

static void gdn_fused_update(int64_t x, int64_t xpitch, int64_t wgt, int64_t bias,
                             int64_t cstate, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
                             int state_len_max, int64_t cache_idx, int64_t ci_stride,
                             int64_t num_accepted, int64_t cu, int num_seqs, int num_v_heads,
                             int num_k_heads, int head_k, int head_v, int width,
                             int max_query_len, int64_t q, int64_t k, int64_t v, int64_t a,
                             int64_t b, int64_t ab_stride, int ab_is_bf16, int64_t A_log,
                             int64_t dt_bias, int64_t state, int64_t st_slot, int64_t st_head,
                             int64_t o, int64_t sidx, int64_t sidx_stride, double scale,
                             double softplus_thr, int64_t barrier_cnt, int o_rows,
                             int64_t stream) {
  int rc = r4d_gdn_fused_update_w4k128v128_bf16(
      (const void*)x, xpitch, (const void*)wgt, (const void*)bias, (void*)cstate, cs_seq,
      cs_dim, cs_tok, state_len_max, (const void*)cache_idx, ci_stride,
      (const void*)num_accepted, (const void*)cu, num_seqs, num_v_heads, num_k_heads, head_k,
      head_v, width, max_query_len, (void*)q, (void*)k, (void*)v, (const void*)a,
      (const void*)b, ab_stride, ab_is_bf16, (const void*)A_log, (const void*)dt_bias,
      (void*)state, st_slot, st_head, (void*)o, (const void*)sidx, sidx_stride, (float)scale,
      (float)softplus_thr, (void*)barrier_cnt, o_rows, (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_fused_update: this build is compiled for head dim 128 "
                             "and conv width 4");
  if (rc == -3)
    throw std::runtime_error("r4d gdn_fused_update: needs a zeroed barrier counter");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_fused_update: hip launch error (code " +
                             std::to_string(rc) + ")");
}

static void gdn_gated_rmsnorm(int64_t x, int64_t z, int64_t w, int64_t o, int64_t rows,
                              int64_t xrow, int64_t zrow, int64_t orow, int width, double eps,
                              int act, int64_t stream) {
  int rc = r4d_gdn_gated_rmsnorm_h128_bf16((const void*)x, (const void*)z, (const void*)w,
                                           (void*)o, rows, xrow, zrow, orow, width, (float)eps,
                                           act, (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_gated_rmsnorm_h128_bf16: this build is compiled for rows "
                             "of 128");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_gated_rmsnorm_h128_bf16: hip launch error (code " +
                             std::to_string(rc) + ")");
}

static void gdn_recurrent_update(int64_t q, int64_t k, int64_t v, int64_t a, int64_t b,
                                 int64_t ab_stride, bool ab_is_bf16, int64_t A_log,
                                 int64_t dt_bias, int64_t state, int64_t state_slot_stride,
                                 int64_t state_head_stride, int64_t o, int64_t cu,
                                 int64_t sidx, int64_t sidx_stride, int64_t num_accepted,
                                 int64_t z_gate, int64_t norm_weight, double norm_eps,
                                 int norm_act, int num_seqs, int num_v_heads, int num_k_heads,
                                 int head_k, int head_v, double scale, double softplus_thr,
                                 int64_t stream) {
  int rc = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
      (const void*)q, (const void*)k, (const void*)v, (const void*)a, (const void*)b, ab_stride,
      ab_is_bf16 ? 1 : 0, (const void*)A_log, (const void*)dt_bias, (void*)state,
      state_slot_stride, state_head_stride, (void*)o, (const void*)cu, (const void*)sidx,
      sidx_stride, (const void*)num_accepted, (const void*)z_gate, (const void*)norm_weight,
      (float)norm_eps, norm_act, num_seqs, num_v_heads, num_k_heads, head_k, head_v,
      (float)scale, (float)softplus_thr, (void*)stream);
  if (rc == -1)
    throw std::runtime_error("r4d gdn_recurrent_update_k128_v128_bf16_fp32state: this build is "
                             "compiled for head_k/head_v 128/128");
  if (rc != 0)
    throw std::runtime_error("r4d gdn_recurrent_update_k128_v128_bf16_fp32state: hip launch "
                             "error (code " + std::to_string(rc) + ")");
}

// =================================================================================================
// all-reduce
// =================================================================================================
static std::tuple<int64_t, py::bytes> ar_ipc_alloc_py(int64_t size, bool finegrained) {
  char h[R4D_AR_HANDLE_BYTES];
  int len = 0;
  int64_t p = r4d_ar_ipc_alloc(size, finegrained ? 1 : 0, h, &len);
  return std::make_tuple(p, py::bytes(h, (size_t)len));
}
static int64_t ar_ipc_open_py(const std::string& handle) {
  return r4d_ar_ipc_open(handle.data(), (int)handle.size());
}

// =================================================================================================
// registry and selector
// =================================================================================================
// select() exists because the geometry of a kernel was being written down three times: in the
// entry point's name, in the check the entry point rejects a mismatch on, and again in whatever
// caller decided it was allowed to make the call. The third copy is the one that drifts -- it
// lives in another repository and nothing ties it back to the kernel. A caller asks the library
// now: name an op and the geometry, get back the kernel that covers it or None, in which case the
// caller keeps whatever it would have used without R4D.
//
// It answers "is there a kernel compiled for this geometry", which is a NECESSARY condition and
// not a sufficient one. Contiguity, strides, alignment and buffer sizes are properties of the
// tensors at a call, not of a model, and stay where they are: in the caller's own check and in the
// entry point.

namespace {

// The spellings a dtype may arrive in, so a caller can pass torch.bfloat16, "torch.bfloat16" or
// "bf16" and mean the same thing.
std::string norm_dtype(const std::string& s) {
  static const struct { const char* from; const char* to; } kMap[] = {
      {"torch.bfloat16", "bf16"},            {"bfloat16", "bf16"},
      {"torch.float16", "fp16"},             {"float16", "fp16"},
      {"half", "fp16"},                      {"torch.float32", "fp32"},
      {"float32", "fp32"},                   {"float", "fp32"},
      {"torch.float8_e4m3fn", "fp8_e4m3"},   {"float8_e4m3fn", "fp8_e4m3"},
      {"torch.float8_e4m3fnuz", "fp8_e4m3"}, {"float8_e4m3fnuz", "fp8_e4m3"},
      {"fp8", "fp8_e4m3"},
  };
  for (const auto& m : kMap) {
    if (s == m.from) return m.to;
  }
  return s;
}

struct Val {
  bool is_num = false;
  long long i = 0;
  std::string s;
};

Val to_val(py::handle h) {
  Val v;
  // bool before int: py::isinstance<py::int_> is true for bools, and `causal=True` has to mean 1.
  if (py::isinstance<py::bool_>(h)) {
    v.is_num = true;
    v.i = h.cast<bool>() ? 1 : 0;
  } else if (py::isinstance<py::int_>(h)) {
    v.is_num = true;
    v.i = h.cast<long long>();
  } else {
    v.s = norm_dtype(py::str(h).cast<std::string>());  // str() also catches torch.dtype
  }
  return v;
}

bool set_contains(const char* set, const std::string& v) {
  const std::string s(set);
  for (size_t p = 0; p <= s.size();) {
    size_t e = s.find(' ', p);
    if (e == std::string::npos) e = s.size();
    if (e > p && s.compare(p, e - p, v) == 0) return true;
    p = e + 1;
  }
  return false;
}

std::string show(const Val& v) { return v.is_num ? std::to_string(v.i) : v.s; }

// True when every constraint holds. Otherwise `why` names the one that did not and what it wanted:
// "no kernel" on its own is not an answer anyone can act on.
bool matches(const R4DKernelInfo* k, const py::kwargs& kw, std::string* why) {
  for (int i = 0; i < k->n_constraints; ++i) {
    const R4DConstraint& c = k->constraints[i];
    if (!kw.contains(c.key)) {
      *why = std::string(c.key) + " not given";
      return false;
    }
    const Val v = to_val(kw[c.key]);
    if (c.op == R4D_C_IN) {
      if (v.is_num || !set_contains(c.sval, v.s)) {
        *why = std::string(c.key) + "=" + show(v) + ", needs one of: " + c.sval;
        return false;
      }
      continue;
    }
    if (!v.is_num) {
      *why = std::string(c.key) + "=" + v.s + " is not a number";
      return false;
    }
    bool ok = false;
    const char* rel = "";
    switch (c.op) {
      case R4D_C_EQ:  ok = (v.i == c.ival); rel = "== "; break;
      case R4D_C_LE:  ok = (v.i <= c.ival); rel = "<= "; break;
      case R4D_C_GE:  ok = (v.i >= c.ival); rel = ">= "; break;
      case R4D_C_DIV: ok = (c.ival != 0 && v.i % c.ival == 0); rel = "divisible by "; break;
      default: break;
    }
    if (!ok) {
      *why = std::string(c.key) + "=" + std::to_string(v.i) + ", needs " + rel +
             std::to_string(c.ival);
      return false;
    }
  }
  *why = "ok";
  return true;
}

py::list constraint_list(const R4DKernelInfo* k) {
  static const char* kOpNames[] = {"eq", "le", "ge", "divisible_by", "in"};
  py::list out;
  for (int i = 0; i < k->n_constraints; ++i) {
    const R4DConstraint& c = k->constraints[i];
    py::dict d;
    d["key"] = c.key;
    d["op"] = (c.op >= 0 && c.op <= R4D_C_IN) ? kOpNames[c.op] : "?";
    if (c.op == R4D_C_IN) {
      d["value"] = py::str(c.sval);
    } else {
      d["value"] = py::int_(c.ival);
    }
    out.append(d);
  }
  return out;
}

py::dict kernel_dict(const R4DKernelInfo* k) {
  py::dict d;
  d["name"] = k->name;
  d["family"] = k->family;
  d["op"] = k->op;
  d["computes"] = k->computes;
  d["shape"] = k->shape;
  d["dtypes"] = k->dtypes;
  d["constraints"] = constraint_list(k);
  return d;
}

// Every distinct question select() has been asked in this process, in the order first asked, with
// the answer it got. A caller binds its entry points once at import and then never mentions the
// registry again, so by the time anything is ready to report -- after the model is loaded and the
// graphs are captured -- the questions would otherwise be gone. They are the interesting half:
// the geometry in them is the geometry of the model actually loaded, and a None answer is
// the record of a kernel this build deliberately did not use.
//
// Kept here rather than in a wrapper on the Python side because this is the one function every
// caller passes through, whichever module it lives in and whenever it happens to import. Repeats
// are folded rather than appended -- one attention layer asks per instantiation, so a 64-layer
// model asks the same question 64 times -- and the count is kept, because "asked 64 times" says
// how much of the model resolved to that kernel. select() runs under the GIL, so no lock.
struct Selection {
  std::string op;
  std::string key;    // the query sorted and flattened, so repeats compare equal whatever the
                      // caller's keyword order was
  std::vector<std::pair<std::string, std::string>> query;  // as asked, so it reads back naturally
  std::string kernel;  // empty means nothing matched
  std::string reason;  // why not, when nothing matched
  long long count = 1;
};

std::vector<Selection>& selection_log() {
  static std::vector<Selection> log;
  return log;
}

// Normalised through the same show(to_val()) select() matches on, so the record reads back as the
// question actually asked: a caller that passed torch.bfloat16 is recorded as bf16. Keyword order
// is kept, because a caller writes the geometry in the order that reads best.
std::vector<std::pair<std::string, std::string>> query_of(const py::kwargs& kw) {
  std::vector<std::pair<std::string, std::string>> q;
  for (auto item : kw) {
    q.emplace_back(py::str(item.first).cast<std::string>(), show(to_val(item.second)));
  }
  return q;
}

std::string query_key(std::vector<std::pair<std::string, std::string>> q) {
  std::sort(q.begin(), q.end());
  std::string key;
  for (const auto& kv : q) key += kv.first + "=" + kv.second + ";";
  return key;
}

void record(const std::string& op, const py::kwargs& kw, const std::string& kernel,
            const std::string& reason) {
  auto q = query_of(kw);
  auto key = query_key(q);
  for (auto& s : selection_log()) {
    if (s.op == op && s.key == key) {  // same question, therefore the same answer
      ++s.count;
      return;
    }
  }
  selection_log().push_back({op, std::move(key), std::move(q), kernel, reason, 1});
}

}  // namespace

static py::list kernels() {
  py::list out;
  for (int i = 0; i < r4d_kernel_count(); ++i) out.append(kernel_dict(r4d_kernel_at(i)));
  return out;
}

// Every op this build implements, mapped to the parameters select() will ask for. A caller that
// does not know what to pass reads this rather than the source.
static py::dict ops() {
  py::dict out;
  for (int i = 0; i < r4d_kernel_count(); ++i) {
    const R4DKernelInfo* k = r4d_kernel_at(i);
    py::list keys = out.contains(k->op) ? out[py::str(k->op)].cast<py::list>() : py::list();
    for (int j = 0; j < k->n_constraints; ++j) {
      const char* key = k->constraints[j].key;
      bool seen = false;
      for (const auto& h : keys) {
        if (h.template cast<std::string>() == key) { seen = true; break; }
      }
      if (!seen) keys.append(key);
    }
    out[py::str(k->op)] = keys;
  }
  return out;
}

static py::object select_kernel(const std::string& op, const py::kwargs& kw) {
  std::vector<std::string> misses;
  for (int i = 0; i < r4d_kernel_count(); ++i) {
    const R4DKernelInfo* k = r4d_kernel_at(i);
    if (op != k->op) continue;
    std::string why;
    if (matches(k, kw, &why)) {
      record(op, kw, k->name, "");
      return py::str(k->name);
    }
    if (std::find(misses.begin(), misses.end(), why) == misses.end()) misses.push_back(why);
  }
  // A refusal is worth recording too, with the constraint that caused it: it is the reason a
  // fallback is running, and "no kernel" on its own is not something a reader can act on.
  std::string reason;
  if (misses.empty()) {
    reason = "no kernel in this build has op '" + op + "'";
  } else {
    for (size_t i = 0; i < misses.size(); ++i) reason += (i ? "; " : "") + misses[i];
  }
  record(op, kw, "", reason);
  return py::none();
}

// The log select() has been keeping. explain() deliberately does not appear here: it asks nothing,
// it only re-runs a question already recorded.
static py::list selections() {
  py::list out;
  for (const auto& s : selection_log()) {
    py::dict d;
    d["op"] = s.op;
    py::dict q;
    for (const auto& kv : s.query) q[py::str(kv.first)] = kv.second;
    d["query"] = q;
    d["kernel"] = s.kernel.empty() ? py::object(py::none()) : py::object(py::str(s.kernel));
    d["reason"] = s.reason;
    d["count"] = s.count;
    out.append(d);
  }
  return out;
}

// Same walk as select(), but reporting every candidate and why it was or was not chosen.
static py::list explain(const std::string& op, const py::kwargs& kw) {
  py::list out;
  for (int i = 0; i < r4d_kernel_count(); ++i) {
    const R4DKernelInfo* k = r4d_kernel_at(i);
    if (op != k->op) continue;
    std::string why;
    const bool ok = matches(k, kw, &why);
    py::dict d;
    d["name"] = k->name;
    d["ok"] = ok;
    d["reason"] = why;
    out.append(d);
  }
  return out;
}

static py::list constraints_of(const std::string& name) {
  for (int i = 0; i < r4d_kernel_count(); ++i) {
    const R4DKernelInfo* k = r4d_kernel_at(i);
    if (name == k->name) return constraint_list(k);
  }
  throw std::runtime_error("r4d: no kernel named '" + name + "'");
}

// The vision-encoder kernel takes none of the paged machinery, so it does not go through
// mk_args: pointers, the cu_seqlens segmentation and the shape, and that is all.
static void attn_vit(int64_t q, int64_t k, int64_t v, int64_t out, int64_t cu_seqlens,
                     int num_seqs, int max_seqlen, int heads, int head_dim, double scale,
                     int64_t stream) {
  const int rc = r4d_attn_vit_h72_bf16((const void*)q, (const void*)k, (const void*)v, (void*)out,
                                       (const void*)cu_seqlens, num_seqs, max_seqlen, heads,
                                       head_dim, (float)scale, (void*)stream);
  check(rc, "r4d attn_vit_h72_bf16");
}

PYBIND11_MODULE(r4d, m) {
  m.doc() = "R4D: gfx1201 (RDNA4) kernels -- attention, gated delta net, all-reduce, skinny GEMM. "
            "Entry points are named <family>_<op>_<compiled-in geometry>; see kernels().";
  m.attr("__version__") = R4D_VERSION;

  // ---- attention. One function per (phase, KV cache dtype); the name is the kernel. ----
  m.def("attn_prefill_h256_gqa6_fp8kv",  &attn<0, 0>);
  m.def("attn_prefill_h256_gqa6_bf16kv", &attn<0, 1>);
  m.def("attn_decode_h256_gqa6_fp8kv",   &attn<1, 0>);
  m.def("attn_decode_h256_gqa6_bf16kv",  &attn<1, 1>);
  m.def("attn_decode_h256_gqa6_scratch_bytes", &attn_decode_scratch_bytes);
  m.def("attn_vit_h72_bf16", &attn_vit);
  {
    int hd = 0, gqa = 0, bs = 0, rows = 0;
    r4d_attn_dims(&hd, &gqa, &bs, &rows);
    m.attr("ATTN_HEAD_DIM")        = hd;
    m.attr("ATTN_GQA")             = gqa;
    m.attr("ATTN_BLOCK_SIZE")      = bs;    // not in the names: 16 for the whole library
    m.attr("ATTN_MAX_DECODE_ROWS") = rows;  // q_len * gqa the decode kernel can hold
    int vhd = 0, vrl = 0, vrs = 0, vsp = 0;
    r4d_attn_vit_dims(&vhd, &vrl, &vrs, &vsp);
    m.attr("ATTN_VIT_HEAD_DIM")   = vhd;
    m.attr("ATTN_VIT_ROWS_LARGE") = vrl;
    m.attr("ATTN_VIT_ROWS_SMALL") = vrs;
    m.attr("ATTN_VIT_SPLIT")      = vsp;    // segment length at which the two arms swap
  }

  // ---- gated delta net ----
  m.def("gdn_chunk_scan_k128_v128_c64_bf16", &gdn_chunk_scan);
  m.def("gdn_kkt_solve_k128_c64_bf16", &gdn_kkt_solve);
  m.def("gdn_conv_prep_w4_h128_bf16", &gdn_conv_prep);
  m.def("gdn_conv_update_w4_h128_bf16", &gdn_conv_update);
  m.def("gdn_fused_update_w4k128v128_bf16", &gdn_fused_update);
  m.def("gdn_recurrent_update_k128_v128_bf16_fp32state", &gdn_recurrent_update);
  m.def("gdn_gated_rmsnorm_h128_bf16", &gdn_gated_rmsnorm);
  {
    int hk = 0, hv = 0, c = 0;
    r4d_gdn_dims(&hk, &hv, &c);
    m.attr("GDN_HEAD_K") = hk;
    m.attr("GDN_HEAD_V") = hv;
    m.attr("GDN_CHUNK")  = c;
  }

  // ---- all-reduce (one-shot, 2 ranks, P2P) ----
  m.def("ar_ipc_alloc", &ar_ipc_alloc_py, py::arg("size"), py::arg("finegrained") = true);
  m.def("ar_ipc_open", &ar_ipc_open_py);
  m.def("ar_ipc_free", &r4d_ar_ipc_free);
  m.def("ar_ipc_memzero", &r4d_ar_ipc_memzero);
  m.def("ar_ipc_enable_peer", &r4d_ar_ipc_enable_peer);
  m.def("ar_oneshot_2rank_exact", &r4d_ar_oneshot_2rank_exact);
  m.def("ar_oneshot_2rank_exact_nq", &r4d_ar_oneshot_2rank_exact_nq);
  m.def("ar_oneshot_2rank_wht6", &r4d_ar_oneshot_2rank_wht6);
  m.def("ar_oneshot_3rank_exact", &r4d_ar_oneshot_3rank_exact);
  m.def("ar_oneshot_4rank_exact", &r4d_ar_oneshot_4rank_exact);
  {
    int group = 0, bits = 0, chunk_elems = 0;
    r4d_ar_wht6_dims(&group, &bits, &chunk_elems);
    m.attr("AR_MAX_BLOCKS")       = r4d_ar_max_blocks();
    m.attr("AR3_MAX_BLOCKS")      = r4d_ar_3rank_max_blocks();
    m.attr("AR4_MAX_BLOCKS")      = r4d_ar_4rank_max_blocks();
    m.attr("AR_WHT6_GROUP")       = group;
    m.attr("AR_WHT6_BITS")        = bits;
    m.attr("AR_WHT6_CHUNK_ELEMS") = chunk_elems;
  }

  // ---- skinny GEMM ----
  m.def("gemm_bf16_nt_m16", &r4d_gemm_bf16_nt_m16, py::arg("a"), py::arg("w"), py::arg("c"),
        py::arg("M"), py::arg("K"), py::arg("N"), py::arg("WV"), py::arg("SK"), py::arg("stream"));
  m.attr("GEMM_MAX_M") = r4d_gemm_bf16_nt_m16_max_m();

  // ---- what this build contains ----
  m.def("kernels", &kernels,
        "Every kernel in this build: name, family, the op it implements, what it computes, the "
        "geometry it is compiled for, its operand dtypes, and that geometry as testable "
        "constraints.");
  m.def("ops", &ops,
        "The ops this build implements, each mapped to the parameter names select() will ask "
        "for.");
  m.def("select", &select_kernel, py::arg("op"),
        "select(op, **geometry) -> kernel name, or None when this build has nothing for that "
        "geometry and the caller should use whatever it would have used without R4D. Answers "
        "whether a kernel EXISTS for a shape; contiguity, strides and buffer sizes are per-call "
        "and remain the caller's check.");
  m.def("explain", &explain, py::arg("op"),
        "explain(op, **geometry) -> every candidate for that op with whether it matched and, when "
        "it did not, which constraint failed.");
  m.def("constraints", &constraints_of, py::arg("name"),
        "The constraints one kernel is compiled for, as {key, op, value} rows.");
  m.def("selections", &selections,
        "Every distinct select() this process has been asked, as {op, query, kernel, reason, "
        "count} rows, in the order first asked. kernel is None where nothing matched, and reason "
        "names the constraint that refused it.");
}
