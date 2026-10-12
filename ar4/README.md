# 4-rank one-shot all-reduce for Radiance (TP=4 on 4x R9700)

Written 2026-10-11. **How to build, install and run it: [HOWTO.md](HOWTO.md).** Radiance's TP=2 speed comes partly from libr4d's one-shot P2P all-reduce
(`ar_oneshot_2rank_exact`): each card pushes its partial sum straight into the peer's VRAM over
PCIe, flips a flag, reduces locally. At TP=4 that kernel declined (`custom AR disabled: no exact
all-reduce kernel for world_size=4`) and every reduction rode RCCL. This directory adds the 4-rank
version.

## Measured (ar_bench.py, bf16 rows x 5120, HIP-graph replay, us per call)

| bytes | RCCL ws=2 | r4d 2-rank | RCCL ws=4 | **r4d 4-rank** |
|---:|---:|---:|---:|---:|
| 10 KB (1 row) | 20 | 5 | 35 | **6** |
| 80 KB (8 rows = DFlash verify) | 24 | 7 | 38 | **13** |
| 160 KB | 25 | 8 | 41 | **18** |
| 320 KB | 30 | 12 | 48 | **30** |
| 640 KB | 39 | 19 | 62 | **57** |
| 1.25 MB | 56 | 33 | 85 | 104 |
| 40 MB | 1200 | 965 | 1650 | 3730 |

The 4-rank push moves 3x the message out of each card, so it wins below ~700 KB and loses above.
`RADIANCE_AR4_MAX_KB` (default 768) caps it; prefill messages (80 MiB at chunk 8192) stay on RCCL.
Block/thread geometry sweep (min_nb 4..24, nt 256..1024) was all within noise; defaults kept.
Correctness: every rank holds identical bits (gathered and compared) and the sum matches an fp32
reference. Not bit-identical to RCCL at 4 ranks (RCCL's ring sums in a different order); the 2-rank
path is unchanged and still bit-identical to RCCL.

## Files

- `r4d_ar_oneshot_4rank_exact.hip` — the kernel, templated on the 3-rank one: three receive
  regions x two seq-parity slots, three flag rows, canonical fp32 sum `((x0+x1)+x2)+x3`.
- `libr4d-ar4.patch` — the full diff against libr4d `b9e42ab-rx6` (header decl, pybind, registry
  row with `world_size == 4`, build.sh unit). On monolith the built tree is
  `~/.cache/radiance-libr4d/b9e42ab-rx6-ar4` (`make IMAGE=stilldeadcode/vllm-radiance:0.9.3`).
- `patch_ar_4rank.py` — radiance_allreduce.py patch (same style as the repo's patch_ar_3rank.py,
  which it supersedes): peer list / scratch regions follow world size, 3- and 4-rank launch arms,
  ws=4 message cap, install gate ws in (2,3,4). Lives at `~/vllm-mxfp4/patch_ar_4rank.py`.
- `run_paroquant_ar4.diff` — the two-line change to the launcher: run the patch, pass
  `RADIANCE_USE_R4D_AR` / `RADIANCE_AR4_MAX_KB` through. `~/vllm-mxfp4/paroquant/run_paroquant_ar4.sh`.
- `tp4-ar4-launch.sh`, `tp4-ar4-trial.sh` — TP=4 serve on :8090 with the new library, then
  `bench_oai.py` + `prefill_conc.py`. `~/bench/` on monolith.
- `ar_bench.py`, `ar-bench.sh` — the microbenchmark (`PATCH=1 R4D=<dir> ar-bench.sh 0,1,2,3`).

## Serving results (2026-10-12, `bench_oai.py` medium effort; `prefill_conc.py` aggregate)

| TP=4 int5 + DFlash2 | long-pp / prose / code tok/s | 1 / 4 / 8 streams |
|---|---|---|
| RCCL (any boot) | 194 / 128 / 157 | 139 / 401 / 608 |
| one-shot, slow dispatch state (6 of 8 boots) | 139 / 108 / 125 | 115 / 345 / 499 |
| **one-shot + `GPU_MAX_HW_QUEUES=1`** (2/2 boots) | **210 / 164 / 191** | **170 / 448 / 650** |
| 10-09 TP=4 baseline (RCCL, rx6) | 179 / 116 / 159 | 118 / 390 / 595 |

Decode step (single stream, 8-wide verify): one-shot 18.6 ms, RCCL 22.3 ms, one-shot in the slow
state 28.4 ms. The slow state is the gfx1201 dispatch-gap behaviour of ROCm#6347 / legacy-rocm-build#6685
(see the project work log); `GPU_MAX_HW_QUEUES=1` is now the default in `run_paroquant_ar4.sh`.

Diagnostics left in `patch_ar_4rank.py`, all off by default: `RADIANCE_AR4_SHADOW=1` (eager serve: RCCL
reference + cross-rank bit check on every call), `RADIANCE_AR4_MIN_KB`, `RADIANCE_AR4_SKIP_FILES=<regex>`
(route call sites to RCCL at graph capture), `RADIANCE_AR4_NB` (fixed block count, default 8).
Other files here: `tp4-boot.sh` (boot + wait), `pair-hwq.sh` (the deciding A/B), `prof-step.sh` +
`trace_summary.py` + `step_breakdown.py` (torch-profiler capture and per-step analysis),
`start-keeper-container.sh` (nexus-style idle-context keeper, container form; not needed with the flag).
