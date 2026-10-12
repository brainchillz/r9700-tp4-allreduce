// R4D vision-encoder attention: dense, non-causal, head_dim 72, bf16 in and out.
//
// This is the same transposed core as the paged prefill kernel -- S^T = K.Q^T, so the wave32
// column-distributed accumulator gives each lane exactly ONE query row and the softmax state is
// lane-private. That choice is worth more here than it is at head_dim 256, because it is what
// removes the QK-C-fragment -> PV-A-fragment transpose of P through LDS: the C fragment of S^T
// (lane holds query `c`, keys 8h+i) IS the B fragment the PV matmul wants (P^T, key rows,
// query columns). A formulation that carries that LDS round trip instead does not get past torch
// SDPA for this shape.
//
// head_dim 72 is carried natively as 5 k-tiles of 16 = 80, with dims 72..79 structurally zero:
// the Q fragment of the last tile is zeroed on the h==1 half, the last 8 columns of every staged
// K row are zeroed once, and the last tile's output store is taken only by the h==0 half (the
// h==1 half would run off the end of the 72-element row). 80/72 = 11% of padded WMMA work, the
// same overhead the 64+16 Triton split pays.
//
// What differs from the paged kernel, beyond the geometry:
//   * K and V are contiguous [tokens, heads, 72] bf16, so there is no block table, no descale and
//     no fp8 widening -- the whole two-round-trip staging problem that cost the paged kernel 20%
//     is simply absent. K is fetched 16 bytes per thread; V is transposed into LDS with
//     global_load_tr_b128 exactly as there.
//   * Non-causal: no per-row key limit. Only the last key tile is masked, and the branch is
//     workgroup-uniform rather than per-wave.
//   * Varlen over segments: one launch covers every image in the batch (grid.z), so the caller
//     does not split by cu_seqlens and concatenate.
//
// The 16-bit operand format is f16, for the reason in r4d_dt16.h: gfx1201 has no bf16 convert, so
// the softmax's f32 -> 16-bit pack is 1 instruction per pair instead of ~6. A bf16 operand build
// is kept reachable (V_O_F16 off) because it makes the K/V staging a pure byte copy -- it loses,
// but the arm is what proves the trade rather than assuming it.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

// Measured optimisation steps, each separable so any one can be ablated after a geometry change.
#define V_O_F16        1   // f16 rather than bf16 as the 16-bit operand format
#define V_O_FOLDQ      8   // scale * log2(e) folded into the query once at load
#define V_O_PF2     1024   // fragment prefetch depth: 2
#define V_O_PF4     2048   //                          4
#define V_O_SGB     4096   // hard scheduling barrier between the prefetch group and its WMMAs
#define V_O_DOT2   16384   // v_dot2_f32_f16 row sum off the packed P
#define V_O_GPREV 1048576  // next tile's V fetched into registers across the m-tile loop
#define V_O_LDSB  2097152  // LDS-scoped barrier instead of __syncthreads()
// GPREK: the same register prefetch for K. Unlike the paged kernel -- which sits at 240 VGPRs and
// spills any second prefetch, so a wait lands on it immediately and it stops being a prefetch --
// this geometry runs at ~130, and the K fetch is only three uint4. Without it the tile's global
// load is issued between the two barriers with only the V LDS stores to cover it.
#define V_O_GPREK 4194304
// MIXFMT: bf16 operands for QK, f16 for PV. The two matmuls have opposite conversion costs. K and
// V arrive as bf16 and are staged through LDS, so an f16 LDS carries a bf16 -> f16 convert per
// element (3 VALU per dword) on the staging path; but P is produced in f32 by the softmax every
// m-tile, and f32 -> bf16 is ~6 VALU of software RTNE against ONE v_cvt_pkrtz_f16_f32. Splitting
// the formats takes the cheap side of both: K is copied into LDS as raw bf16 with no conversion
// at all and the query is converted once at load, while V and P stay f16. Accuracy is unchanged --
// K and V are bf16 sources, so both widenings are exact.
#define V_O_MIXFMT 8388608

// Pricing probes, wrong on purpose. A probe that changes the DATAFLOW measures nothing (making p
// constant makes s dead and deletes the QK loop), so each of these deletes work while leaving every
// value's producer-consumer chain intact. Count the emitted instructions before believing a number.
#ifndef V_ABL
#define V_ABL 0
#endif
#define V_ABL_EXP    1   // the 8 v_exp_f32 per m-tile become the identity
#define V_ABL_STAGE  2   // no K/V fetch or LDS store inside the tile loop (barriers kept)
#define V_ABL_MAX    4   // no running-max test and no lazy rescale
#define V_ABL_LDS    8   // every K/V fragment read once before the tile loop and reused

// bf16 dword -> 16-bit dword in the target format, with the query scale folded in.
template<class D, int FOLDQ, int F16>
__device__ __forceinline__ uint32_t r4d_vit_qcvt(uint32_t w, float s) {
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(__builtin_bit_cast(float, w << 16) * s,
                 __builtin_bit_cast(float, w & 0xffff0000u) * s);
}

template<int NWARPS, int TILE, int QB, int OPT>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_vit_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                         const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                         const int* __restrict__ cu, int heads, float scale)
{
    constexpr int F16   = (OPT & V_O_F16)   ? 1 : 0;
    constexpr int FOLDQ = (OPT & V_O_FOLDQ) ? 1 : 0;
    constexpr int PF    = (OPT & V_O_PF4) ? 4 : ((OPT & V_O_PF2) ? 2 : 1);
    constexpr int SGB   = (OPT & V_O_SGB)   ? 1 : 0;
    constexpr int DOT2  = ((OPT & V_O_DOT2) && F16) ? 1 : 0;
    constexpr int GPREV = (OPT & V_O_GPREV) ? 1 : 0;
    constexpr int LDSB  = (OPT & V_O_LDSB)  ? 1 : 0;
    constexpr int ABL_EXP   = (V_ABL & V_ABL_EXP)   ? 1 : 0;
    constexpr int ABL_STAGE = (V_ABL & V_ABL_STAGE) ? 1 : 0;
    constexpr int ABL_MAX   = (V_ABL & V_ABL_MAX)   ? 1 : 0;
    constexpr int ABL_LDS   = (V_ABL & V_ABL_LDS)   ? 1 : 0;
    constexpr int GPREK = (OPT & V_O_GPREK) ? 1 : 0;
    constexpr int MIXF  = (OPT & V_O_MIXFMT) ? 1 : 0;
    typedef DT16<F16> D;                       // V and P: the format the softmax packs into
    typedef DT16<MIXF ? 0 : F16> DK;           // K and Q: bf16 under MIXFMT, else the same
    typedef typename D::frag  frag16;
    typedef typename DK::frag fragK;

    constexpr int HD    = 72;               // the real head dim
    constexpr int HDP   = 80;               // padded to whole 16-wide WMMA k-tiles
    constexpr int NKS   = HDP / 16;         // 5
    constexpr int DGRP  = HD / 8;           // 9 eight-element dim groups spanning the real dims
    constexpr int ROWS  = NWARPS * 16 * QB;
    constexpr int MT    = TILE / 16;
    constexpr int NTHR  = NWARPS * 32;
    // Row strides padded by 8 shorts: 16-byte aligned (a 16-byte LDS access at an 8-byte-aligned
    // address splits back into two b64) and off the bank period, which puts both fragment reads
    // at the 8-distinct-bank-group minimum.
    constexpr int KSTR  = HDP + 8;
    constexpr int VSTR  = TILE + 8;
    constexpr int KGRP  = TILE / 8;         // eight-key groups per tile; a power of two
    constexpr int NTR   = KGRP * DGRP;      // transposed V units, one per 8 lanes
    constexpr int LGRP  = NTHR / 8;
    constexpr int VPASS = (NTR + LGRP - 1) / LGRP;
    constexpr int KIA   = TILE * 8;         // K fetch items, dims 0..63 (16 B each)
    constexpr int KIB   = TILE;             // K fetch items, dims 64..71
    constexpr int KPA   = (KIA + NTHR - 1) / NTHR;
    constexpr int KPB   = (KIB + NTHR - 1) / NTHR;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;
    static_assert((TILE & (TILE - 1)) == 0 && TILE >= 16, "TILE must be a power of two >= 16");

    __shared__ __align__(16) uint16_t sK[TILE * KSTR];
    __shared__ __align__(16) uint16_t sV[HDP * VSTR];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int qb = blockIdx.x, hd = blockIdx.y, seq = blockIdx.z;

    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    if (slen <= 0 || qb * ROWS >= slen) return;

    const long hstride = (long)heads * HD;          // elements between consecutive tokens
    const float mul = scale * 1.44269504089f;

    // ---- the query fragments, held for the life of the workgroup -----------------------------
    // A warp owns QB tiles of 16 query rows rather than one. Every K and V fragment read out of
    // LDS is then consumed by QB WMMAs instead of one, which is the point: at QB = 1 this kernel
    // issues ten ds_load_b128 per ten WMMAs, and LDS delivers less per cycle than the matrix
    // units consume. The accumulator is only 8 VGPRs per output tile, so the second query tile is
    // affordable where it would not be at head_dim 256.
    int  qpos[QB];
    bool live[QB];
    fragK qf[QB][NKS];
    #pragma unroll
    for (int b = 0; b < QB; ++b) {
        qpos[b] = qb * ROWS + (warp * QB + b) * 16 + c;
        live[b] = qpos[b] < slen;
        const int qrow = live[b] ? qpos[b] : slen - 1;
        const uint16_t* qp = gq + (long)(s0 + qrow) * hstride + (long)hd * HD;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            // The padded half of the last k-tile: dims 72..79 do not exist. Zeroing the query
            // side is not enough on its own -- the K side is zeroed too, because a WMMA
            // contracts, and NaN * 0 would poison the whole score row rather than one element.
            if (t == NKS - 1 && h == 1) { qf[b][t] = DK::mk(make_uint2(0, 0), make_uint2(0, 0)); continue; }
            const uint4 u = *(const uint4*)(qp + 16 * t + 8 * h);
            qf[b][t] = DK::mk(make_uint2(r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.x, mul),
                                         r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.y, mul)),
                              make_uint2(r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.z, mul),
                                         r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.w, mul)));
        }
    }

    // The padded lanes of the staging buffers, written once: staging only ever fills dims 0..71.
    #pragma unroll 1
    for (int i = tid; i < TILE; i += NTHR) *(uint4*)&sK[i * KSTR + HD] = make_uint4(0, 0, 0, 0);
    #pragma unroll 1
    for (int i = tid; i < (HDP - HD) * VSTR; i += NTHR) sV[HD * VSTR + i] = 0;

    float m_ref[QB], soff[QB], l_i[QB];
    v8f acc[QB][NKS];
    #pragma unroll
    for (int b = 0; b < QB; ++b) {
        m_ref[b] = ABL_MAX ? 0.0f : -3.0e38f;
        soff[b]  = ABL_MAX ? PSHIFT : (PSHIFT + 3.0e38f);
        l_i[b]   = 0.0f;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) acc[b][t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    }

    // ---- staging, split into a global FETCH (to registers) and an LDS STORE -------------------
    const int vj = tid & 7, lgi = tid >> 3;
    uint4 kregA[KPA], kregB[KPB], vreg[VPASS];

    auto fetchK = [&](int k0) {
        #pragma unroll
        for (int p = 0; p < KPA; ++p) {
            const int i = tid + p * NTHR;
            if (KIA % NTHR == 0 || i < KIA) {
                const int key = i >> 3, g = i & 7;
                const int kk = (k0 + key < slen) ? (k0 + key) : (slen - 1);
                kregA[p] = *(const uint4*)(gk + (long)(s0 + kk) * hstride + (long)hd * HD + g * 8);
            }
        }
        #pragma unroll
        for (int p = 0; p < KPB; ++p) {
            const int i = tid + p * NTHR;
            if (KIB % NTHR == 0 || i < KIB) {
                const int kk = (k0 + i < slen) ? (k0 + i) : (slen - 1);
                kregB[p] = *(const uint4*)(gk + (long)(s0 + kk) * hstride + (long)hd * HD + 64);
            }
        }
    };

    auto storeK = [&]() {
        #pragma unroll
        for (int p = 0; p < KPA; ++p) {
            const int i = tid + p * NTHR;
            if (KIA % NTHR == 0 || i < KIA) {
                const uint4 r = kregA[p];
                *(uint4*)&sK[(i >> 3) * KSTR + (i & 7) * 8] =
                    make_uint4(DK::from_bf16w(r.x), DK::from_bf16w(r.y),
                               DK::from_bf16w(r.z), DK::from_bf16w(r.w));
            }
        }
        #pragma unroll
        for (int p = 0; p < KPB; ++p) {
            const int i = tid + p * NTHR;
            if (KIB % NTHR == 0 || i < KIB) {
                const uint4 r = kregB[p];
                *(uint4*)&sK[i * KSTR + 64] =
                    make_uint4(DK::from_bf16w(r.x), DK::from_bf16w(r.y),
                               DK::from_bf16w(r.z), DK::from_bf16w(r.w));
            }
        }
    };

    // V is read transposed straight out of global memory: the eight lanes of a group each name a
    // different key row at the same 8 dims, and global_load_tr_b128 hands lane j the column for
    // dim d0+j across all eight of them -- one contiguous 16-byte LDS row of V^T per lane.
    auto fetchV = [&](int k0) {
        #pragma unroll
        for (int r = 0; r < VPASS; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u & (KGRP - 1), dg = u / KGRP;
                const int kk0 = k0 + kg * 8 + vj;
                const int kk = (kk0 < slen) ? kk0 : (slen - 1);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(
                    gv + (long)(s0 + kk) * hstride + (long)hd * HD + dg * 8));
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VPASS; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u & (KGRP - 1), dg = u / KGRP;
                const uint4 v = vreg[r];
                *(uint4*)&sV[(dg * 8 + vj) * VSTR + kg * 8] =
                    make_uint4(D::from_bf16w(v.x), D::from_bf16w(v.y),
                               D::from_bf16w(v.z), D::from_bf16w(v.w));
            }
        }
    };

    fragK kb0; frag16 vb0;
    if (ABL_LDS) {
        const uint4 u = *(const uint4*)(&sK[c * KSTR] + 8 * h);
        kb0 = DK::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
        const uint4 w = *(const uint4*)(&sV[c * VSTR] + 8 * h);
        vb0 = D::mk(make_uint2(w.x, w.y), make_uint2(w.z, w.w));
    }

    const int ntiles = (slen + TILE - 1) / TILE;
    if (GPREV && !ABL_STAGE) fetchV(0);
    if (GPREK && !ABL_STAGE) fetchK(0);

    for (int ti = 0; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        const bool tail = (k0 + TILE > slen);
        if (LDSB) lds_barrier(); else __syncthreads();
        if (!ABL_STAGE) {
            if (!GPREK) fetchK(k0);
            if (!GPREV) fetchV(k0);
            storeV();          // V first: already in registers, so it covers K's load wait
            storeK();
        }
        if (LDSB) lds_barrier(); else __syncthreads();
        // Issued here, before the WMMAs below; the wait that consumes it is past the next barrier.
        if (GPREV && !ABL_STAGE && ti + 1 < ntiles) fetchV(k0 + TILE);
        if (GPREK && !ABL_STAGE && ti + 1 < ntiles) fetchK(k0 + TILE);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s[QB];
            #pragma unroll
            for (int b = 0; b < QB; ++b) s[b] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    fragK kb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        if (g + j < NKS) {
                            if (ABL_LDS) { kb[j] = kb0; continue; }
                            const uint4 u = *(const uint4*)(krow + 16 * (g + j));
                            kb[j] = DK::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        if (g + j < NKS)
                            #pragma unroll
                            for (int b = 0; b < QB; ++b) s[b] = DK::wmma(kb[j], qf[b][g + j], s[b]);
                }
            }

            frag16 p16[QB];
            #pragma unroll
            for (int b = 0; b < QB; ++b) {
                float sc[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[b][e] : s[b][e] * mul;

                // Non-causal: only the last tile can hold keys past the end of the segment, and
                // the test is workgroup-uniform rather than per-row.
                if (tail) {
                    const int kbase = k0 + mt * 16 + 8 * h;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) sc[e] = (kbase + e < slen) ? sc[e] : -INFINITY;
                }

                float smax = sc[0];
                #pragma unroll
                for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
                if (!ABL_MAX && wave_any(smax > m_ref[b] + PGROW)) {
                    const float rmax  = fmaxf(smax, swap16(smax));
                    const float mnew  = fmaxf(fmaxf(m_ref[b], rmax), -3.0e38f);
                    const float alpha = ex2(m_ref[b] - mnew);
                    #pragma unroll
                    for (int dt = 0; dt < NKS; ++dt)
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) acc[b][dt][e] *= alpha;
                    l_i[b] *= alpha;
                    m_ref[b] = mnew;
                    soff[b]  = PSHIFT - mnew;
                }

                float pv[8], lsum = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float p = ABL_EXP ? (sc[e] + soff[b]) : ex2(sc[e] + soff[b]);
                    pv[e] = p;
                    if (!DOT2) lsum += p;
                }
                if (!DOT2) l_i[b] += lsum;

                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pv[e], pv[e + 1]);
                if (DOT2) {
                    const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i[b] = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one,
                                                        l_i[b], false);
                }
                // QK left lane (c,h) holding keys 8h+e of query c. That IS the B fragment of
                // P^T for the PV matmul, so no exchange and no LDS round trip.
                p16[b] = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));
            }

            {
                const uint16_t* vbase = &sV[c * VSTR + mt * 16] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        if (g + j < NKS) {
                            if (ABL_LDS) { vb[j] = vb0; continue; }
                            const uint4 u = *(const uint4*)(vbase + (g + j) * 16 * VSTR);
                            vb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        if (g + j < NKS)
                            #pragma unroll
                            for (int b = 0; b < QB; ++b)
                                acc[b][g + j] = D::wmma(vb[j], p16[b], acc[b][g + j]);
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < QB; ++b) {
        if (!live[b]) continue;
        const float ls = l_i[b] + swap16(l_i[b]);
        const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
        uint16_t* op = go + (long)(s0 + qpos[b]) * hstride + (long)hd * HD;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) {
            // The last tile covers dims 64..79; the h==1 half of it is padding and would store
            // past the end of a 72-element row.
            if (dt == NKS - 1 && h == 1) continue;
            uint32_t tw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2)
                tw[e >> 1] = (uint32_t)f32_to_bf16(acc[b][dt][e] * inv)
                           | ((uint32_t)f32_to_bf16(acc[b][dt][e + 1] * inv) << 16);
            *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
        }
    }
}

// ---- entry point -----------------------------------------------------------------------------
// Two instantiations, chosen on the segment length. Both are the same kernel; they differ only in
// how many query rows a workgroup owns. ROWS 256 is the argmin from 1K patches upward, but it
// quantises badly below that -- at 576 patches it leaves a workgroup two thirds dead and measures
// 1.33x over the Triton kernel where ROWS 128 measures 1.76x. The crossover sits between 576 and
// 1024; 768 is the midpoint and neither arm is far off at that length.
//
// Every geometry constant is overridable so a sweep build is one -D away, and V_SPLIT = 0 forces
// every length through the large arm.
#ifndef V_NWARPS
#define V_NWARPS 16          // large arm: ROWS = 256
#endif
#ifndef V_TILE
#define V_TILE 128
#endif
#ifndef V_QB
#define V_QB 1
#endif
#ifndef V_SNWARPS
#define V_SNWARPS 8          // small arm: ROWS = 128
#endif
#ifndef V_STILE
#define V_STILE 64
#endif
#ifndef V_SQB
#define V_SQB 1
#endif
#ifndef V_SPLIT
#define V_SPLIT 768
#endif
// F16 | FOLDQ | PF4 | SGB | DOT2 | GPREV | LDSB | GPREK. MIXFMT is off: it measured identical, and
// one operand format and one WMMA variant is the simpler thing to keep.
#ifndef V_OPT
#define V_OPT 7362569
#endif

// q, k, v, o are [total_tokens, heads, 72] contiguous bf16; cu_seqlens is [num_seqs + 1] int32 and
// bounds one image (or one attention window) per entry, so a whole batch is one launch. max_seqlen
// is the host-side bound on those lengths -- the kernel reads the device-side cu_seqlens itself.
extern "C" int r4d_attn_vit_h72_bf16(const void* q, const void* k, const void* v, void* o,
                                     const void* cu_seqlens, int num_seqs, int max_seqlen,
                                     int heads, int head_dim, float scale, void* stream)
{
    if (head_dim != 72) return -1;
    if (heads <= 0) return -2;
    if (num_seqs <= 0 || max_seqlen <= 0) return 0;

#define V_LAUNCH(NW, TL, QBS)                                                                  \
    do {                                                                                       \
        constexpr int ROWS = (NW) * 16 * (QBS);                                                \
        dim3 grid((max_seqlen + ROWS - 1) / ROWS, heads, num_seqs);                             \
        hipLaunchKernelGGL((r4d_attn_vit_kernel<(NW), (TL), (QBS), V_OPT>), grid,               \
                           dim3((NW) * 32), 0, (hipStream_t)stream, (const uint16_t*)q,         \
                           (const uint16_t*)k, (const uint16_t*)v, (uint16_t*)o,                \
                           (const int*)cu_seqlens, heads, scale);                               \
    } while (0)

    if (V_SPLIT > 0 && max_seqlen < V_SPLIT) V_LAUNCH(V_SNWARPS, V_STILE, V_SQB);
    else                                     V_LAUNCH(V_NWARPS,  V_TILE,  V_QB);
#undef V_LAUNCH

    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - (int)e;
    return 0;
}

// The geometry this build is compiled for: head_dim, the two query-block heights, and the length
// at which the launcher switches between them.
extern "C" void r4d_attn_vit_dims(int* head_dim, int* rows_large, int* rows_small, int* split) {
    if (head_dim)    *head_dim    = 72;
    if (rows_large)  *rows_large  = V_NWARPS * 16 * V_QB;
    if (rows_small)  *rows_small  = V_SNWARPS * 16 * V_SQB;
    if (split)       *split       = V_SPLIT;
}
