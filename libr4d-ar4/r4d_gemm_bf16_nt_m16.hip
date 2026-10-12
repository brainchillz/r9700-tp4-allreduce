// r4d_gemm_bf16_nt_m16.hip - r4d_gemm_bf16_nt_m16: skinny bf16 GEMM on gfx1201 / RDNA4 (R9700).
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 throughout -- a torch Linear, weight stored (N,K), which is the
// "nt" in the name. M <= 16 is the constraint that matters: the accumulators are held per compile-
// time M, and at these shapes the weight read dominates and rocBLAS does poorly. K and N are
// runtime. Pure HIP behind the C ABI in r4d.h (no torch/extension.h).
//
// The MoE router gate (M<=16, K=2048, N=256, a ~1 MB weight) is what it was tuned on and is its
// only caller today, but nothing here is router-specific: any small bf16 projection in this band
// fits. Hence the generic name -- the caller's role does not belong in a kernel's name.
//
// Design:
//  * Split-K within a block: block = WV columns x SK k-splits waves, so WV*SK waves per block fill the
//    machine (one wave per column leaves the 32 CUs under-filled). The SK partials are reduced through a
//    small LDS scratch in-block - no separate reduce kernel, so no extra launch latency - and the reduction
//    is deterministic, which is what keeps a router -> top-k on top of it reproducible.
//  * A is read straight from L2 (no LDS staging); the weight is streamed with 128-bit loads issued up-front
//    for maximum in-flight, and the inner product uses the packed v_dot2_f32_bf16 builtin.
#include "r4d.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#ifndef WAVE
#define WAVE 32          // wave32 (RDNA4); -DWAVE=64 (+ -mwavefrontsize64) for wave64
#endif
#define VEC 8            // bf16 per 128-bit (dwordx4) load
#define R4D_GEMM_MAX_M 16          // max supported M (=n)
typedef int int4v __attribute__((ext_vector_type(4)));      // native 128-bit vector
typedef __bf16 bf162 __attribute__((ext_vector_type(2)));   // packed bf16 pair for v_dot2_f32_bf16
// weight load: cache-resident by default, since the gate weights fit the 64 MB Infinity Cache and
// that measured faster than streaming them. -DR4D_ROUTER_TEMPORAL=0 restores the non-temporal
// (stream, read-once) load.
#ifndef R4D_GEMM_TEMPORAL
#define R4D_GEMM_TEMPORAL 1
#endif
#if R4D_GEMM_TEMPORAL
#define WLOAD(p) (*(p))
#else
#define WLOAD(p) __builtin_nontemporal_load(p)
#endif

template <int MT>
__global__ void r4d_gemm_bf16_nt_m16_kernel(const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
                      __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
  extern __shared__ float red[];                       // WV*SK*MT SK-reduction scratch
  int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
  int col_local = wave / SK, ksplit = wave % SK;
  int j = blockIdx.x * WV + col_local;
  int Ks = K / SK, kbase = ksplit * Ks;
  float acc[MT];
  #pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  if (j < N) {
    const __hip_bfloat16* Wj = W + (size_t)j * K + kbase + lane * VEC;
    int niter = Ks / (WAVE * VEC);
    int4v wbuf[8];
    #pragma unroll
    for (int t = 0; t < 8; t++)                          // issue all weight loads up-front (max in-flight)
      if (t < niter) wbuf[t] = WLOAD((const int4v*)(Wj + t * WAVE * VEC));
    for (int t = 0; t < niter; t++) {
      const bf162* wp = (const bf162*)&wbuf[t];
      int kk = kbase + lane * VEC + t * WAVE * VEC;
      #pragma unroll
      for (int i = 0; i < MT; i++) {
        int4 av = *(const int4*)(A + (size_t)i * K + kk); const bf162* ap = (const bf162*)&av;
        #pragma unroll
        for (int p = 0; p < VEC / 2; p++)
          acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
      }
    }
    #pragma unroll
    for (int i = 0; i < MT; i++) { float v = acc[i];
      for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) red[(col_local * SK + ksplit) * MT + i] = v; }
  }
  __syncthreads();
  if (ksplit == 0 && j < N && lane < MT) {              // SK-reduce + write one column of C
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col_local * SK + s) * MT + lane];
    C[(size_t)lane * N + j] = (__hip_bfloat16)v;
  }
}

// grid = ceil(N/WV) blocks; block = WV*SK waves. Compile-time-M dispatch keeps exactly M accumulators.
void r4d_gemm_bf16_nt_m16(long a, long w, long c, int M, int K, int N, int WV, int SK,
                          long stream) {
  if (M > R4D_GEMM_MAX_M) throw std::runtime_error("r4d gemm_bf16_nt_m16: M must be <= 16");
  if (K % SK != 0 || (K / SK) % (WAVE * VEC) != 0) throw std::runtime_error("bad SK");
  if (WV * SK * WAVE > 1024) throw std::runtime_error("threads/block>1024");
  dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
  size_t smem = (size_t)WV * SK * M * sizeof(float);
  auto A = (const __hip_bfloat16*)a; auto Wp = (const __hip_bfloat16*)w; auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  #define LAUNCH(MT) hipLaunchKernelGGL(r4d_gemm_bf16_nt_m16_kernel<MT>, grid, block, smem, st, A, Wp, Cp, K, N, WV, SK)
  switch (M) {
    case 1: LAUNCH(1); break;   case 2: LAUNCH(2); break;   case 3: LAUNCH(3); break;   case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break;   case 6: LAUNCH(6); break;   case 7: LAUNCH(7); break;   case 8: LAUNCH(8); break;
    case 9: LAUNCH(9); break;   case 10: LAUNCH(10); break; case 11: LAUNCH(11); break; case 12: LAUNCH(12); break;
    case 13: LAUNCH(13); break; case 14: LAUNCH(14); break; case 15: LAUNCH(15); break; case 16: LAUNCH(16); break;
    default: throw std::runtime_error("M unsupported");
  }
  #undef LAUNCH
}

int r4d_gemm_bf16_nt_m16_max_m(void) { return R4D_GEMM_MAX_M; }
