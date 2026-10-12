## Summary

Two changes that together take TP=4 decode on four R9700s from **194 / 128 / 157** to **210 / 164 / 191 tok/s** (`bench_oai.py` long-pp / prose / code, int5 PARO Qwen3.8-27B + DFlash2, 211 W caps), and aggregate 1 / 4 / 8 streams from 139 / 401 / 608 to **170 / 448 / 650**.

1. **A four-rank version of the r4d one-shot all-reduce**, shipped as libr4d extras revision **rx11** (= rx6 + `r4d_ar_oneshot_4rank_exact`). At TP=4 today the 2-rank kernel declines (`custom AR disabled: no exact all-reduce kernel for world_size=4`) and every one of the ~130 reductions per decode step rides RCCL at ~38 µs. The 4-rank push kernel does the same call in ~13 µs and takes a decode step's all-reduce share from 4.65 ms to 1.6 ms.
2. **`GPU_MAX_HW_QUEUES=1` in both launchers.** Without it a Radiance process on gfx1201 lands at spawn, at random, in a slow dispatch state that costs more than the kernel saves (ROCm/legacy-rocm-build#6347 and #6685). More on that below, because it will bite anyone measuring this stack.

## What is in the change

| file | change |
|---|---|
| `r4d_radiance_extras_rx11.patch` | rx6 + the 4-rank kernel: `r4d_ar_oneshot_4rank_exact.hip` (templated on the 3-rank kernel), declaration in `r4d.h`, pybind in `r4d_module.hip`, registry row with `world_size == 4`, build unit in `build.sh`. Applies clean on upstream `b9e42ab` and builds in the 0.9.3 image. |
| `patch_ar_4rank.py` | replaces `patch_ar_3rank.py` in the serve chain (never run both): peer list, scratch regions and launch follow the world size (2, 3, 4); at ws=4 a message cap `RADIANCE_AR4_MAX_KB` (768) and a fixed block count `RADIANCE_AR4_NB` (8); the compressed wht6 path stays 2-rank only; `RADIANCE_AR4_SHADOW=1` is a correctness gate (eager serve: recompute every call with RCCL, all-gather and compare across ranks). |
| `serve-mxfp4.sh` | `TP=4` selects rx11; the chain runs `patch_ar_4rank.py`; `GPU_MAX_HW_QUEUES` (default 1) and the `RADIANCE_AR4_*` knobs pass through. |
| `paroquant/run_paroquant.sh` | same env passthroughs and the patch in its chain; `RADIANCE_USE_R4D_AR` now overridable (it is the A/B switch); the "libr4d not built" hint names rx11 for TP=4. |
| `PERFORMANCE.md`, `README.md` | the results row, a TP=4 section, the shadow-check knob under Diagnostics. |

Nothing in the 2-rank path changes. `RADIANCE_USE_R4D_AR=0` puts the all-reduce back on RCCL with no other difference.

## The kernel

Same design as the 2- and 3-rank kernels: each block pushes its chunk of the local input into all three peers' IPC scratch (stores interleaved, one `s_wait_storecnt` drain), releases one flag per peer with a system-scope store, spins on its own three flags, then reduces local input plus the three regions in **canonical rank order** `((x0 + x1) + x2) + x3` in fp32, rounded once. The rank only picks which operand is the local input, so all four ranks hold identical bits, which is the invariant the fused epilogues rely on. Scratch per rank is 3 receive regions × 2 seq-parity slots × `max_bytes`; sender *i*'s region in receiver *j* is `i - (i > j)`. Per-block device-resident sequence counters keep it graph-safe exactly as before.

It is **not bit-identical to RCCL** at four ranks: RCCL's ring sums in a different order, in bf16. This one is fp32-accumulated, so it is the more exact of the two. The 2-rank kernel keeps its RCCL bit-identity.

Why the cap: the push moves the message to three peers, so the one-shot wins below ~700 KB and loses above. Measured at 4 ranks under HIP graph replay (µs per call, RCCL → one-shot): 10 KB 35 → 6, 80 KB 38 → 13, 320 KB 48 → 30, 640 KB 57 → 62, 1.25 MB 85 → 104. Decode messages are 80 KB for an 8-wide verify; prefill chunks stay on RCCL.

## Correctness

- Microbenchmark: every output matches an fp32 reference and all ranks hold identical bits, including with a layer-sized GEMM between calls, deliberate rank skew, host traffic, and alternating message sizes under graph replay.
- In situ: an eager TP=4 serve with `RADIANCE_AR4_SHADOW=1` over **38,500 calls**: 0 outside bf16 tolerance (max relative deviation from RCCL 1.2e-2, which is the summation-order difference), **0 cross-rank bit differences**. Outputs are coherent at temperature 0 and 0.7; DFlash acceptance is unchanged at ~3.6 tokens per step.
- Profiler (torch, FULL graphs): per decode step the all-reduce kernels total 1.63 ms vs 4.65 ms for RCCL at the same 129 calls, with every other category unchanged.

**Not run:** BetterBench paired A/B and the GSM8K gate on this stack. Please run your own gates before merging.

## The dispatch state, since it will confuse anyone benchmarking this

Six of eight TP=4 boots of the one-shot configuration came up slow: 139 / 108 / 125 tok/s, 28 ms per step instead of 18.6. Same binary, same launcher, same model. The profiler shows identical kernel durations in both states; the whole difference is the idle gap after every small kernel inside graph replay, ~3 µs in a good boot and ~15 µs in a bad one, times ~1,000 kernels per step. That is the gfx1201 behaviour in ROCm/legacy-rocm-build#6347 (bimodal R9700 decode, fixed at spawn) and #6685 (pending `hipStreamWaitEvent` on other hardware queues adds ~17 µs per kernel, gone with `GPU_MAX_HW_QUEUES` ≤ 2). With `GPU_MAX_HW_QUEUES=1` every boot was fast; concurrency 1 to 8 showed no cost; the RCCL configuration was unchanged by the flag (it happened to land fast every time on this box). Box: 4x R9700 (0x7551) on PCIe 5 x16, MES firmware 0x91, amdgpu-dkms 6.19.14 on kernel 7.0.0-38, ROCm 7.14 in the 0.9.3 image, KFD reports 6 queues per worker process at TP=2.

I made the flag the default in both launchers rather than TP=4-only because the issue is per process, not per TP width, and the upstream threads report no cost either.

## How to try it

```sh
TP=4 PREPARE_ONLY=1 ./serve-mxfp4.sh         # builds b9e42ab-rx11 (a few minutes)
TP=4 GPUS=0,1,2,3 ./serve-mxfp4.sh
# ParoQuant: TP=4 GPUS=0,1,2,3 R4D_KEY=b9e42ab-rx11 ./paroquant/run_paroquant.sh
```

Boot log should show `custom all-reduce INSTALLED (rank=N max=768KB ... nb=8..8)` on all four workers. `RADIANCE_STEP_TRACE=50` is a quick way to see which dispatch state a server landed in: single-stream `gpu_span` ≈ 18.5 ms is good, ≈ 28 ms is the slow state.

## Full write-up and tooling

Microbenchmarks, the profiler analysis scripts, the shadow-check serve, raw logs and a build/operations HOWTO are in https://github.com/brainchillz/r9700-tp4-allreduce.
