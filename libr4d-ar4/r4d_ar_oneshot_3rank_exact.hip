// r4d_ar_oneshot_3rank_exact.hip: r4d_ar_oneshot_3rank_exact -- multi-block one-shot P2P-BAR
// all-reduce for EXACTLY THREE ranks (three R9700 over PCIe; TP=3 via the radiance dummy-head
// padding). Templated on r4d_ar_oneshot_2rank_exact, which stays byte-identical; everything
// below is additive.
//
// What generalises from the 2-rank kernel, and what does not:
//  * PUSH model, unchanged: each rank writes ITS input into BOTH peers' IPC scratch, then reduces
//    (local input + the two peer messages now in local scratch) -> out. Under full-duplex PCIe
//    this symmetric one-shot ties a hub/tree on wire time at every size (2S out and 2S in on the
//    slow link, in parallel, vs S up then S down sequentially), so the algorithm is kept and no
//    topology-aware tree is needed.
//  * SCRATCH: two receive REGIONS (one per sender) x two seq-parity SLOTS x max_bytes. In
//    receiver j's scratch, sender i writes region i - (i > j): region 0 always holds the
//    lower-ranked peer, region 1 the higher. The host computes "my region in each peer" once.
//  * FLAGS: two regions x AR_MAX_BLOCKS, one flag per (sender, block), same layout as the regions.
//  * HANDSHAKE: thread 0 releases both peer flags, then spins until BOTH of its own flags for
//    this block reach the sequence number. Per-block device-resident seq counters and the
//    (seq & 1) double-buffer are unchanged; the reuse argument carries over per peer pair: a rank
//    at seq s+2 has passed the s+1 handshake with both peers, and a peer releases flag s+1 only
//    after its kernel s (which read slot s) has completed in stream order.
//  * REDUCE: a CANONICAL rank-ascending fp32 sum ((x0 + x1) + x2) on every rank. Two-term fp32
//    addition is commutative, which is what gave the 2-rank kernel cross-rank bit-identity for
//    free; with three terms the association order has to be fixed explicitly, and it is: the rank
//    selects the operand mapping, not the summation order. The result is rounded to T once, so
//    all three ranks hold the same bits -- the replicated-state invariant the fused epilogues
//    depend on.
//  * drain=3 (s_wait_storecnt) drains BOTH peers' pushes at once: the two stores of each chunk
//    are interleaved, and PCIe keeps posted writes to the same peer ordered, so each peer's
//    release flag lands after that peer's data.
#include "r4d.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define R4D_AR3_SPIN_MAX 4000000000ULL
#define R4D_AR3_MAX_BLOCKS 512   // == R4D_AR_MAX_BLOCKS of the 2-rank units (flags per region)

namespace {

__device__ __forceinline__ void store_sys_rel3(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq3(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float to_f3(const float& x) { return x; }
__device__ __forceinline__ float to_f3(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float to_f3(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void from_f3(float v, float& o) { o = v; }
__device__ __forceinline__ void from_f3(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void from_f3(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// out = ((a + b) + c) per lane in fp32, rounded to T once. The caller passes the operands in
// canonical rank order, so every rank evaluates the identical expression.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd3(const int4& a, const int4& b, const int4& c) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  const T* pc = reinterpret_cast<const T*>(&c);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) from_f3((to_f3(pa[j]) + to_f3(pb[j])) + to_f3(pc[j]), pr[j]);
  return r;
}

template <typename T, int LANES>
__global__ void r4d_ar_oneshot_3rank_exact_kernel(
    const int4* __restrict__ in4, int4* __restrict__ out4,
    const int4* my_base, int4* peer_lo_base, int4* peer_hi_base,
    unsigned int* my_flags, unsigned int* peer_lo_flags, unsigned int* peer_hi_flags,
    unsigned int* seq_ctrs, int n16, int slot_stride16, int region_stride16,
    int my_region_in_lo, int my_region_in_hi, int rank, int drain, int acq) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;  // per-block, replay-safe
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s & 1u);
  // where MY data goes in each peer, and where each peer's data lands in MY scratch
  int4* p_lo = peer_lo_base + (size_t)my_region_in_lo * region_stride16 + (size_t)slot * slot_stride16;
  int4* p_hi = peer_hi_base + (size_t)my_region_in_hi * region_stride16 + (size_t)slot * slot_stride16;
  const int4* r0 = my_base + (size_t)0 * region_stride16 + (size_t)slot * slot_stride16;  // lower peer
  const int4* r1 = my_base + (size_t)1 * region_stride16 + (size_t)slot * slot_stride16;  // higher peer

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // push chunk -> both peers, stores interleaved so one drain covers both
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      p_lo[k] = v0; p_hi[k] = v0;
      p_lo[k + nt] = v1; p_hi[k + nt] = v1;
      p_lo[k + 2 * nt] = v2; p_hi[k + 2 * nt] = v2;
      p_lo[k + 3 * nt] = v3; p_hi[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) { int4 v = in4[k]; p_lo[k] = v; p_hi[k] = v; }
  }
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel3(&peer_lo_flags[my_region_in_lo * R4D_AR3_MAX_BLOCKS + b], s);
    store_sys_rel3(&peer_hi_flags[my_region_in_hi * R4D_AR3_MAX_BLOCKS + b], s);
    const unsigned int* f0 = &my_flags[0 * R4D_AR3_MAX_BLOCKS + b];
    const unsigned int* f1 = &my_flags[1 * R4D_AR3_MAX_BLOCKS + b];
    unsigned long long z = 0;
    bool d0 = false, d1 = false;
    while (!(d0 && d1)) {
      if (!d0) d0 = load_sys_acq3(f0) >= s;
      if (!d1) d1 = load_sys_acq3(f1) >= s;
      if (++z > R4D_AR3_SPIN_MAX) break;
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  // reduce chunk in canonical rank order: x0 + x1 + x2 where x_rank is this rank's own input.
  //   rank 0: (in + r0) + r1      rank 1: (r0 + in) + r1      rank 2: (r0 + r1) + in
  // fp32 two-term addition is commutative, so the first two cases are one expression.
  {
    int k = start + tid;
    if (rank == 2) {
      for (; k + 3 * nt < end; k += 4 * nt) {
        out4[k]          = vadd3<T, LANES>(r0[k],          r1[k],          in4[k]);
        out4[k + nt]     = vadd3<T, LANES>(r0[k + nt],     r1[k + nt],     in4[k + nt]);
        out4[k + 2 * nt] = vadd3<T, LANES>(r0[k + 2 * nt], r1[k + 2 * nt], in4[k + 2 * nt]);
        out4[k + 3 * nt] = vadd3<T, LANES>(r0[k + 3 * nt], r1[k + 3 * nt], in4[k + 3 * nt]);
      }
      for (; k < end; k += nt) out4[k] = vadd3<T, LANES>(r0[k], r1[k], in4[k]);
    } else {
      for (; k + 3 * nt < end; k += 4 * nt) {
        out4[k]          = vadd3<T, LANES>(in4[k],          r0[k],          r1[k]);
        out4[k + nt]     = vadd3<T, LANES>(in4[k + nt],     r0[k + nt],     r1[k + nt]);
        out4[k + 2 * nt] = vadd3<T, LANES>(in4[k + 2 * nt], r0[k + 2 * nt], r1[k + 2 * nt]);
        out4[k + 3 * nt] = vadd3<T, LANES>(in4[k + 3 * nt], r0[k + 3 * nt], r1[k + 3 * nt]);
      }
      for (; k < end; k += nt) out4[k] = vadd3<T, LANES>(in4[k], r0[k], r1[k]);
    }
  }
}

}  // namespace

// Host entry. Scratch per rank is 2 regions x 2 slots x (slot_stride16 * 16) bytes; flags per
// rank are 2 x R4D_AR_MAX_BLOCKS uints. peer_lo / peer_hi are the peers with the lower / higher
// rank than `rank`; the region this rank writes in each is derived here (i - (i > j)).
void r4d_ar_oneshot_3rank_exact(long my_scratch, long peer_lo_scratch, long peer_hi_scratch,
                                long my_flags, long peer_lo_flags, long peer_hi_flags,
                                long seq_ctrs, long slot_stride16, long rank,
                                long inp, long out, long n_elem, long dtype,
                                long stream_i, long nblocks, long nthreads,
                                long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  if (rank < 0 || rank > 2) throw std::runtime_error("all_reduce_3rank: rank must be 0, 1 or 2");
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce_3rank: n_elem*esize not 16B-aligned");
  if (nbytes / 16 > slot_stride16) throw std::runtime_error("all_reduce_3rank: message exceeds a scratch slot");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR3_MAX_BLOCKS) nb = R4D_AR3_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  // rank r's peers in ascending order are lo < hi; my region in receiver j is r - (r > j).
  const int lo = (rank == 0) ? 1 : 0;
  const int hi = (rank == 2) ? 1 : 2;
  const int my_region_in_lo = (int)rank - (rank > lo ? 1 : 0);
  const int my_region_in_hi = (int)rank - (rank > hi ? 1 : 0);
  const int region_stride16 = 2 * (int)slot_stride16;
#define L3(T, LN) r4d_ar_oneshot_3rank_exact_kernel<T, LN><<<nb, nt, 0, st>>>(                  \
    (const int4*)inp, (int4*)out, (const int4*)my_scratch, (int4*)peer_lo_scratch,               \
    (int4*)peer_hi_scratch, (unsigned*)my_flags, (unsigned*)peer_lo_flags,                       \
    (unsigned*)peer_hi_flags, (unsigned*)seq_ctrs, n16, (int)slot_stride16, region_stride16,     \
    my_region_in_lo, my_region_in_hi, (int)rank, (int)drain, (int)acq)
  if (dtype == 0) L3(__hip_bfloat16, 8);
  else if (dtype == 1) L3(__half, 8);
  else if (dtype == 2) L3(float, 4);
  else throw std::runtime_error("all_reduce_3rank: bad dtype");
#undef L3
}

int r4d_ar_3rank_max_blocks(void) { return R4D_AR3_MAX_BLOCKS; }
