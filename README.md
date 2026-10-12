# A 4-rank one-shot all-reduce for Radiance/vLLM on 4x Radeon AI PRO R9700

Tensor-parallel decode on four R9700s (gfx1201) spends its time in all-reduces. Radiance (the
[`vllm-mxfp4`](https://github.com/jdrews/vllm-mxfp4) fork of vLLM for RDNA4) replaces RCCL with a one-shot
P2P push kernel from its [libr4d](https://codeberg.org/StillDeadcode/libr4d) library, but only for two ranks.
This repo adds the **four-rank kernel**, the wiring to use it, the benchmarks that measured it, and the
diagnosis of a **gfx1201 dispatch-latency state** that hid the gain until `GPU_MAX_HW_QUEUES=1` was set.

| TP=4, int5 Qwen3.8-27B + DFlash2, 4x R9700 at 211 W | long-prompt / prose / code tok/s | 1 / 4 / 8 streams |
|---|---|---|
| RCCL all-reduce | 194 / 128 / 157 | 139 / 401 / 608 |
| **4-rank one-shot + `GPU_MAX_HW_QUEUES=1`** | **210 / 164 / 191** | **170 / 448 / 650** |
| 4-rank one-shot, slow dispatch state (6 of 8 boots without the flag) | 139 / 108 / 125 | 115 / 345 / 499 |

Per call at decode sizes (80 KB, HIP graph replay): RCCL ~38 µs, one-shot ~13 µs; a decode step's all-reduce
share drops from 4.65 ms to 1.6 ms. All four ranks hold bit-identical results (fp32 accumulation in canonical
rank order); 38,500 in-situ calls checked against RCCL with zero mismatches.

Upstream: Radiance PR [jdrews/vllm-mxfp4#2](https://github.com/jdrews/vllm-mxfp4/pull/2); the dispatch state is
ROCm [legacy-rocm-build#6347](https://github.com/ROCm/legacy-rocm-build/issues/6347) and
[#6685](https://github.com/ROCm/legacy-rocm-build/issues/6685) (our data is in both threads).

## Layout

| path | what |
|---|---|
| [`ar4/HOWTO.md`](ar4/HOWTO.md) | **start here**: build, install, run, verify, the dispatch-state flag, kernel design notes, adapting |
| [`ar4/README.md`](ar4/README.md) | the measurements and how the slow state was found |
| `ar4/r4d_ar_oneshot_4rank_exact.hip`, `ar4/libr4d-ar4.patch`, `ar4/patch_ar_4rank.py` | the kernel, the libr4d diff, the Radiance Python patch (full version with hunt diagnostics) |
| `ar4/upstream-pr/` | exactly what went upstream: rx11 patch, lean Python patch, launcher/docs diff, PR body, issue comments, the ROCm reproducer and its run here |
| `libr4d-ar4/` | complete libr4d source tree with the kernel, buildable with `make IMAGE=stilldeadcode/vllm-radiance:0.9.3` (see `PROVENANCE.md`) |
| `launchers/` | `run_paroquant_ar4.sh` (Radiance launcher + the patch + env passthroughs), `patch_ar_4rank.py` |
| `bench/` | `ar_bench.py` (RCCL vs one-shot microbenchmark, in-situ and race modes), `bench_oai.py`, `prefill_conc.py`, the A/B and profiling drivers, `step_breakdown.py` / `trace_summary.py` for torch-profiler traces |
| `keeper/` | the idle-context keeper (the single-card workaround that predates the flag) |
| `results/ar4-2026-10-11/` | benchmark logs, shadow-check log, profiler summaries for a fast and a slow boot |

Hardware: 4x R9700 on PCIe 5 x16 (single-socket Threadripper PRO), P2P for every pair, MES firmware 0x91,
amdgpu-dkms 6.19.14 on kernel 7.0.0-38, ROCm 7.14 inside the Radiance 0.9.3 image. Paths in the scripts
assume the Radiance clone at `~/vllm-mxfp4` and libr4d builds under `~/.cache/radiance-libr4d/<key>`.

Neither upstream project ships a LICENSE file; the libr4d tree here is their code plus our additions, kept for
reference and reproduction. Our own additions you may use freely.
