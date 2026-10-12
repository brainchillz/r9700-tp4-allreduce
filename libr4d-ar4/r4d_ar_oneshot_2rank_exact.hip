// r4d_ar_oneshot_2rank_exact.hip: r4d_ar_oneshot_2rank_exact -- multi-block one-shot P2P-BAR
// all-reduce, exactly two ranks, for dual R9700 (gfx1201/RDNA4, PCIe). "One-shot" is the
// algorithm class: each rank pushes its whole input, so the scratch is sized by the full message
// and there is no ring or reduce-scatter stage. "2rank" is a hard constraint, not a tuning
// choice -- the handshake is one peer flag per block, not a tree. Pure HIP behind the C ABI in
// r4d.h (wheel-ROCm ships no rocThrust, which torch/extension.h needs); the caller passes raw
// data_ptr / numel / dtype / stream. r4d_module.hip is the pybind surface.
//
// Design:
//  * PUSH model: each rank writes ITS input into the PEER's IPC scratch, then reduces
//    (local input + peer's data now in MY scratch) -> out. Only scratch+flags are
//    IPC-shared; input/output stay local -> no cudagraph buffer registration needed.
//  * MULTI-BLOCK, per-chunk independent flags (NO grid barrier): block b owns a
//    contiguous chunk of 16B words -> pushes it to peer, sets peer_flags[b], spins on
//    my_flags[b], reduces its chunk. Spreads the push across CUs (PCIe saturation) and
//    the reduce across CUs (HBM saturation): the single-block bandwidth fix.
//  * cudagraph-safe: seq is a DEVICE-resident PER-BLOCK counter the kernel increments.
//    Both ranks execute the identical graph sequence (SPMD) so seq_ctr[b] stays in
//    lockstep across ranks regardless of which batch-size graphs ran.
//  * race-free reuse: scratch DOUBLE-BUFFERED by (seq&1); same-stream serialization +
//    the per-block flag handshake guarantee a slot is drained before reuse.
//  * FINE-GRAINED memory (hipDeviceMallocFinegrained) => uncached => posted straight to
//    the fabric => SCOPE_SYS release/acquire skip the L2 writeback. drain/acq explicit
//    fences are tunable (all-thread system fences; default drain=1/acq=0 on fine-grained).
#include "r4d.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// vectorized 128-bit add: an int4 packs LANES elements of T; fp32-accumulate reduce.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns words [start,end) of the 16B-word
// space and drives its own flag handshake; fully independent, no cross-block sync.
template <typename T, int LANES>
__global__ void r4d_ar_oneshot_2rank_exact_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;  // per-block, replay-safe
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s & 1u);
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // push chunk -> peer slot, 4x-unrolled for memory-level parallelism (PCIe saturation)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = in4[k];
  }
  // drain modes (ordering of the pushes vs the release flag, so the peer sees data first):
  //  0 = none (relies purely on PCIe posted-write ordering + the release flag)
  //  1 = all-thread __threadfence_system  (safe but emits an L2 wb-invalidate per
  //      wavefront (WASTED on fine-grained/uncached memory; floods at high nt*nb)
  //  2 = single-thread __threadfence_system (cheap but only drains wavefront 0's stores)
  //  3 = all-thread store-completion wait (s_wait_storecnt 0): every wavefront drains
  //      its posted writes into the fabric with NO L2 wb-invalidate; PCIe keeps posted
  //      writes to the same peer ordered, so the later release flag lands after the data.
  //      Correct for FINE-GRAINED memory (uncached; coarse would leave data in L2).
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (load_sys_acq(&my_flags[b]) < s) { if (++z > R4D_AR_SPIN_MAX) break; }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  // reduce chunk (vectorized, 4x-unrolled): out = local input + peer input (staged in local scratch)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          ms[k]);
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     ms[k + nt]);
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], ms[k + 2 * nt]);
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], ms[k + 3 * nt]);
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], ms[k]);
  }
}

// ---- radiance extras: all-reduce with a fused decoder-layer epilogue ------------------------
// r4d_ar_oneshot_2rank_exact_nq: the same push/flag/spin all-reduce, restructured ONE BLOCK PER
// ROW so the epilogue's per-row reductions stay block-local. After reducing its row (bf16
// rounded, exactly like the plain exact kernel writes), the block finishes the decoder layer's
// post-AR epilogue in registers: residual add (fp32), Gemma rms_norm (weight is (1+w), variance
// over the fp32 sum), bf16 round, per-token e4m3 quant with scale = max(amax/448, 1/(448*512)).
// Precision contract matches vllm.ir.ops.fused_add_rms_norm + the radiance traced quant; the
// standalone radiance_add_rms_quant kernel (radiance_mxfp4_fp8.hip) is bit-identical to that
// reference, and this is the same arithmetic on the same rounded values. The reduced y is NOT
// written anywhere -- the epilogue is its only consumer, which saves the out-stream entirely.
// Both ranks compute identical q/scale/res_out (same summation order), so rank parity holds.
// Shares scratch/flags/seq with the plain exact kernel; per-block seq counters stay in rank
// lockstep because both ranks issue identical call sequences (SPMD), grid size included.
#define R4D_ARNQ_T 256
#define R4D_ARNQ_MAXG 5
__global__ __launch_bounds__(R4D_ARNQ_T) void r4d_ar_oneshot_2rank_exact_nq_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base,
    const __hip_bfloat16* __restrict__ res, const __hip_bfloat16* __restrict__ w,
    unsigned char* __restrict__ q, float* __restrict__ scale,
    __hip_bfloat16* __restrict__ res_out,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int k16, int slot_stride16, float eps, int drain, int acq) {
  const int b = blockIdx.x, tid = threadIdx.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int sq = s_seq;
  const int slot = (int)(sq & 1u);
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const int start = b * k16, end = start + k16;
  const int K = k16 * 8;
  {
    int k = start + tid;
    for (; k + 3 * R4D_ARNQ_T < end; k += 4 * R4D_ARNQ_T) {
      int4 v0 = in4[k], v1 = in4[k + R4D_ARNQ_T], v2 = in4[k + 2 * R4D_ARNQ_T],
           v3 = in4[k + 3 * R4D_ARNQ_T];
      ps[k] = v0; ps[k + R4D_ARNQ_T] = v1; ps[k + 2 * R4D_ARNQ_T] = v2;
      ps[k + 3 * R4D_ARNQ_T] = v3;
    }
    for (; k < end; k += R4D_ARNQ_T) ps[k] = in4[k];
  }
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], sq);
    unsigned long long z = 0;
    while (load_sys_acq(&my_flags[b]) < sq) { if (++z > R4D_AR_SPIN_MAX) break; }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  // reduce this row into registers (bf16-rounded like the plain kernel) + start the epilogue
  res += (size_t)b * K; q += (size_t)b * K; res_out += (size_t)b * K;
  float sv[R4D_ARNQ_MAXG][8];
  float ssq = 0.f;
  int ng = 0;
  for (int k = start + tid; k < end; k += R4D_ARNQ_T, ++ng) {
    int4 red = vadd<__hip_bfloat16, 8>(in4[k], ms[k]);
    const __hip_bfloat16* rh = reinterpret_cast<const __hip_bfloat16*>(&red);
    const int e0 = (k - start) * 8;
    int4 rv = ((const int4*)res)[k - start];
    const __hip_bfloat16* vh = reinterpret_cast<const __hip_bfloat16*>(&rv);
    int4 ro;
    __hip_bfloat16* oh = reinterpret_cast<__hip_bfloat16*>(&ro);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = (float)rh[j] + (float)vh[j];
      sv[ng][j] = v;
      oh[j] = (__hip_bfloat16)v;
      ssq += v * v;
    }
    ((int4*)res_out)[k - start] = ro;
    (void)e0;
  }
  __shared__ float lds[R4D_ARNQ_T / 32];
  for (int o = 16; o; o >>= 1) ssq += __shfl_down(ssq, o);
  if ((tid & 31) == 0) lds[tid >> 5] = ssq;
  __syncthreads();
  if (tid < R4D_ARNQ_T / 32) {
    float v = lds[tid];
    for (int o = R4D_ARNQ_T / 64; o; o >>= 1) v += __shfl_down(v, o);
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(lds[0] / (float)K + eps);
  __syncthreads();
  float amax = 0.f;
  int mg = 0;
  for (int k = start + tid; k < end; k += R4D_ARNQ_T, ++mg) {
    int4 wv = ((const int4*)w)[k - start + 0] ; // w indexed by column within the row
    const __hip_bfloat16* wh = reinterpret_cast<const __hip_bfloat16*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float nv = (float)(__hip_bfloat16)(sv[mg][j] * inv * ((float)wh[j] + 1.f));
      sv[mg][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < R4D_ARNQ_T / 32) {
    float v = lds[tid];
    for (int o = R4D_ARNQ_T / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[b] = sc;
  mg = 0;
  for (int k = start + tid; k < end; k += R4D_ARNQ_T, ++mg) {
    unsigned int lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float bq = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, bq, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float bq = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, bq, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    ((uint2*)q)[k - start] = make_uint2(lo, hi);
  }
}

void r4d_ar_oneshot_2rank_exact_nq(long peer_scratch, long my_scratch, long peer_flags,
                                   long my_flags, long seq_ctrs, long slot_stride16,
                                   long inp, long residual, long norm_w, long q_out,
                                   long scale_out, long res_out, long m_rows, long k_cols,
                                   double eps, long stream_i, long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  if (k_cols % 8 != 0) throw std::runtime_error("ar_exact_nq: K must be a multiple of 8");
  if (k_cols > 8 * R4D_ARNQ_T * R4D_ARNQ_MAXG)
    throw std::runtime_error("ar_exact_nq: K too large for the register budget");
  if (m_rows > R4D_AR_MAX_BLOCKS) throw std::runtime_error("ar_exact_nq: too many rows");
  const int k16 = (int)(k_cols / 8);
  r4d_ar_oneshot_2rank_exact_nq_kernel<<<(int)m_rows, R4D_ARNQ_T, 0, st>>>(
      (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch,
      (const __hip_bfloat16*)residual, (const __hip_bfloat16*)norm_w,
      (unsigned char*)q_out, (float*)scale_out, (__hip_bfloat16*)res_out,
      (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs,
      k16, (int)slot_stride16, (float)eps, (int)drain, (int)acq);
}

// The IPC handle is returned through a caller-owned buffer rather than a std::string so this
// stays a C ABI; R4D_AR_HANDLE_BYTES is the fixed size the header promises.
static_assert(sizeof(hipIpcMemHandle_t) <= R4D_AR_HANDLE_BYTES, "ipc handle larger than the ABI buffer");

long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len) {
  void* p = nullptr;
  if (finegrained) HIPOK(hipExtMallocWithFlags(&p, size, hipDeviceMallocFinegrained));
  else HIPOK(hipMalloc(&p, size));
  HIPOK(hipMemset(p, 0, size));
  hipIpcMemHandle_t h;
  HIPOK(hipIpcGetMemHandle(&h, p));
  std::memcpy(out_handle, &h, sizeof(h));
  *out_len = (int)sizeof(h);
  return (long)p;
}
long r4d_ar_ipc_open(const char* handle, int len) {
  hipIpcMemHandle_t h;
  if (len != (int)sizeof(h)) throw std::runtime_error("bad ipc handle size");
  std::memcpy(&h, handle, sizeof(h));
  void* p = nullptr;
  HIPOK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess));
  return (long)p;
}
void r4d_ar_ipc_free(long p) { if (p) (void)hipFree((void*)p); }
void r4d_ar_ipc_memzero(long p, long size) { if (p) HIPOK(hipMemset((void*)p, 0, size)); }

// single-process 2-GPU testing only: enable direct peer access (the runtime path uses IPC handles)
void r4d_ar_ipc_enable_peer(long peer) {
  hipError_t e = hipDeviceEnablePeerAccess((int)peer, 0);
  if (e == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); return; }
  if (e != hipSuccess)
    throw std::runtime_error(std::string("enable_peer: ") + hipGetErrorString(e));
}

void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags,
                            long my_flags, long seq_ctrs, long slot_stride16,
                            long inp, long out, long n_elem, long dtype,
                            long stream_i, long nblocks, long nthreads,
                            long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
#define L(T, LN) r4d_ar_oneshot_2rank_exact_kernel<T, LN><<<nb, nt, 0, st>>>(              \
    (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,        \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16,              \
    (int)slot_stride16, (int)drain, (int)acq)
  if (dtype == 0) L(__hip_bfloat16, 8);
  else if (dtype == 1) L(__half, 8);
  else if (dtype == 2) L(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef L
}

