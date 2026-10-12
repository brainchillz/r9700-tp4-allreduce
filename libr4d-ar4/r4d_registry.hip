// r4d_registry.hip: what this build of the library actually contains.
//
// R4D kernels are specialised, and their names say so -- but a name is only useful to a human
// reading the source. This table is the machine-readable form: a caller that wants to know whether
// R4D covers the model it is about to serve reads it, instead of hardcoding a copy of the
// constraints and drifting from them. Adding a kernel means adding a row here; the build fails to
// mean anything otherwise, since this is also what `r4d.kernels()` prints.
//
// Each row states its geometry twice. `shape` is the prose, and lists every constraint the entry
// point rejects on -- including the ones that are NOT in the name because they are the same across
// the whole library (paged block size 16, bf16 query, gfx1201). `constraints` is the same geometry
// as predicates, which is what `r4d.select()` matches a caller against. The predicates are
// deliberately a NECESSARY condition only: a rule that cannot be written as one predicate on one
// named parameter -- the GEMM's K/SK relationship, for instance -- stays in the prose and in the
// entry point, and select() answers the weaker question. Where the two disagree the prose is the
// one to trust and the row is a bug.
//
// `op` is the key callers select on, and it is the one field several rows may share: fp8 and bf16
// KV are two kernels for one op, and so are the exact and the lossy all-reduce. Rows are in
// preference order within an op -- select() returns the first that matches.
#include "r4d.h"

#define C_EQ(k, v)  {k, R4D_C_EQ, (v), nullptr}
#define C_LE(k, v)  {k, R4D_C_LE, (v), nullptr}
#define C_GE(k, v)  {k, R4D_C_GE, (v), nullptr}
#define C_DIV(k, v) {k, R4D_C_DIV, (v), nullptr}
#define C_IN(k, s)  {k, R4D_C_IN, 0, s}

// ---- attention ---------------------------------------------------------------------------------
// `causal` is a constraint, not a note: the paged kernels are causal by construction and the vision
// kernel is dense, and neither can serve the other's masking.
static const R4DConstraint cAttnPrefillFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const R4DConstraint cAttnPrefillBf16[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
// q_len <= 10 is the q_len*gqa <= 64 of the prose, at the gqa 6 this kernel is compiled for.
static const R4DConstraint cAttnDecodeFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const R4DConstraint cAttnDecodeBf16[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const R4DConstraint cAttnVit[] = {
    C_EQ("head_dim", 72), C_EQ("gqa", 1), C_EQ("causal", 0),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};

// ---- gated delta net ---------------------------------------------------------------------------
static const R4DConstraint cGdnChunkScan[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64),
};
static const R4DConstraint cGdnConvPrep[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64),
};
static const R4DConstraint cGdnConvUpdate[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128),
};
static const R4DConstraint cGdnKktSolve[] = {
    C_EQ("head_k", 128), C_EQ("chunk", 64),
};
static const R4DConstraint cGdnRecurrentUpdate[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128),
};
static const R4DConstraint cGdnFusedUpdate[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128),
};
static const R4DConstraint cGdnGatedRmsNorm[] = {
    C_EQ("channels", 128),
};

// ---- all-reduce --------------------------------------------------------------------------------
// `exact` separates the two, and a caller has to say which it wants: a request that does not
// declare it will accept a lossy wire format gets no kernel rather than the quantised one.
static const R4DConstraint cArExact[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
};
static const R4DConstraint cArWht6[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
};
// radiance extras: three ranks, exact only.
static const R4DConstraint cAr3Exact[] = {
    C_EQ("world_size", 3), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
};
// radiance extras: four ranks, exact only.
static const R4DConstraint cAr4Exact[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
};

// ---- GEMM --------------------------------------------------------------------------------------
static const R4DConstraint cGemmNt[] = {
    C_LE("M", 16), C_DIV("K", 256), C_IN("dtype", "bf16"),
};

#define ROW(arr) arr, (int)(sizeof(arr) / sizeof(arr[0]))

static const R4DKernelInfo kKernels[] = {
    {"attn_prefill_h256_gqa6_fp8kv", "attn", "attn_prefill_paged",
     "paged causal attention, query-tiled (prefill / chunked prefill)",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
     ROW(cAttnPrefillFp8)},
    {"attn_prefill_h256_gqa6_bf16kv", "attn", "attn_prefill_paged",
     "paged causal attention, query-tiled (prefill / chunked prefill)",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen",
     "bf16 query, bf16 kv cache, bf16 out",
     ROW(cAttnPrefillBf16)},
    {"attn_decode_h256_gqa6_fp8kv", "attn", "attn_decode_paged",
     "paged causal attention, split-KV (decode / speculative verify)",
     "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
     ROW(cAttnDecodeFp8)},
    {"attn_decode_h256_gqa6_bf16kv", "attn", "attn_decode_paged",
     "paged causal attention, split-KV (decode / speculative verify)",
     "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
     "bf16 query, bf16 kv cache, bf16 out, f16 partials",
     ROW(cAttnDecodeBf16)},
    {"attn_vit_h72_bf16", "attn", "attn_vit",
     "dense non-causal vision-encoder attention, varlen over images / windows",
     "head_dim 72 native (no padding to 128), multi-head (no GQA), contiguous q/k/v/o, "
     "one cu_seqlens segment per image",
     "bf16 query / key / value / out, f16 operands, fp32 accumulate",
     ROW(cAttnVit)},
    {"gdn_chunk_scan_k128_v128_c64_bf16", "gdn", "gdn_chunk_scan",
     "gated delta net chunked scan: WY recompute, state recurrence and output in one kernel",
     "head_k 128, head_v 128, chunk 64, varlen, state resident in WMMA accumulators",
     "bf16 q/k/v/A, fp32 gate/beta/state, bf16 out, fp32 final state",
     ROW(cGdnChunkScan)},
    {"gdn_conv_prep_w4_h128_bf16", "gdn", "gdn_conv_prep",
     "gdn prefill preamble: causal conv + qkv split + qk l2norm + gating + per-chunk cumsum",
     "conv width 4, head_k/head_v 128, chunk 64, varlen, conv state read and rewritten in place",
     "bf16 x / weights / conv state / q,k,v out, fp32 a,b and gate/beta out",
     ROW(cGdnConvPrep)},
    {"gdn_conv_update_w4_h128_bf16", "gdn", "gdn_conv_update",
     "gdn decode conv: rolling speculative window over the conv state cache, writes q/k/v",
     "conv width 4, head_k/head_v 128, state cache width-1 + num_spec, read at accepted offset",
     "bf16 x / weights / conv state / q,k,v out",
     ROW(cGdnConvUpdate)},
    {"gdn_kkt_solve_k128_c64_bf16", "gdn", "gdn_kkt_solve",
     "gdn chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^dg))^-1, gram never in HBM",
     "head_k 128, chunk 64, varlen; one gram per k head, applied to every v head of the group",
     "bf16 k and A out, fp32 beta / gate / gram / inverse",
     ROW(cGdnKktSolve)},
    {"gdn_fused_update_w4k128v128_bf16", "gdn", "gdn_fused_update",
     "gdn decode step as ONE launch: conv update -> grid barrier -> recurrent update",
     "conv width 4, head_k/head_v 128, no fused row norm; needs a zeroed barrier counter",
     "bf16 x / conv state / q,k,v; fp32 recurrent state",
     ROW(cGdnFusedUpdate)},
    {"gdn_recurrent_update_k128_v128_bf16_fp32state", "gdn", "gdn_recurrent_update",
     "gdn decode recurrent update: gating, qk l2norm, delta-rule state update, gated rms norm "
     "and output",
     "head_k 128, head_v 128, paged state, one state written per candidate token",
     "bf16 q/k/v and out, fp32 a,b and state cache",
     ROW(cGdnRecurrentUpdate)},
    {"gdn_gated_rmsnorm_h128_bf16", "gdn", "gdn_gated_rmsnorm",
     "gated rms norm over a row: out = rms(x) . w . act(z), for the prefill path only",
     "128 channels per row, one wave per row; silu or sigmoid gate",
     "bf16 x / z / out, fp32 weight",
     ROW(cGdnGatedRmsNorm)},
    {"ar_oneshot_2rank_exact", "ar", "allreduce",
     "all-reduce (sum), one-shot push over P2P IPC scratch",
     "exactly 2 ranks, peer-accessible; scratch sized by the full message; cudagraph-safe",
     "bf16 / fp16 / fp32 payload, fp32 accumulate, bit-identical across ranks",
     ROW(cArExact)},
    {"ar_oneshot_2rank_wht6", "ar", "allreduce",
     "all-reduce (sum) with a Walsh-Hadamard-rotated 6-bit wire payload -- lossy",
     "exactly 2 ranks, peer-accessible; groups of 64 elements; cudagraph-safe",
     "bf16 / fp16 payload, 6 bits + bf16 scale per group on the wire, fp32 accumulate",
     ROW(cArWht6)},
    {"ar_oneshot_3rank_exact", "ar", "allreduce",
     "all-reduce (sum), one-shot push over P2P IPC scratch, three ranks",
     "exactly 3 ranks, all pairs peer-accessible; scratch holds two receive regions x two slots "
     "of the full message; cudagraph-safe",
     "bf16 / fp16 / fp32 payload, fp32 accumulate in rank order, bit-identical across ranks",
     ROW(cAr3Exact)},
    {"ar_oneshot_4rank_exact", "ar", "allreduce",
     "all-reduce (sum), one-shot push over P2P IPC scratch, four ranks",
     "exactly 4 ranks, all pairs peer-accessible; scratch holds three receive regions x two slots "
     "of the full message; cudagraph-safe; decode-size messages (3x push)",
     "bf16 / fp16 / fp32 payload, fp32 accumulate in rank order, bit-identical across ranks",
     ROW(cAr4Exact)},
    {"gemm_bf16_nt_m16", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ W[N,K]^T, split-K reduced in LDS",
     "M <= 16; K divisible by SK and K/SK by 256; N, K otherwise runtime",
     "bf16 A / W / C, fp32 accumulate, deterministic reduction",
     ROW(cGemmNt)},
};

extern "C" int r4d_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}

extern "C" const R4DKernelInfo* r4d_kernel_at(int i) {
    if (i < 0 || i >= r4d_kernel_count()) return nullptr;
    return &kKernels[i];
}
