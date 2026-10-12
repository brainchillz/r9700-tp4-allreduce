# HOWTO: the 4-rank one-shot all-reduce for Radiance/vLLM on 4x Radeon AI PRO R9700

This is the implementation and operations guide. The measurements and the story of how it was found
are in `README.md` (this directory).

## 1. What it is, in one paragraph

Radiance (the `vllm-mxfp4` fork of vLLM for gfx1201) replaces RCCL for tensor-parallel all-reduces with
a **one-shot P2P push kernel** from its `libr4d` library: every rank writes its partial sum straight into
the peers' VRAM over PCIe (IPC-mapped, fine-grained memory), flips a per-block flag, waits for the
peers' flags, and reduces locally. That kernel existed for exactly 2 ranks (and a 3-rank variant for a
TP=3 experiment). At TP=4 it declined and every one of the ~130 reductions per decode step went through
RCCL at ~38 µs. `r4d_ar_oneshot_4rank_exact` is the 4-rank version: ~13 µs per call at decode sizes,
3 ms saved per decode step, bit-identical results on all four ranks.

A second, unrelated thing is needed to actually see the gain on this GPU: `GPU_MAX_HW_QUEUES=1`
(section 6). Without it a Radiance process lands at random in a slow dispatch state that costs more than
the kernel saves.

## 2. Prerequisites

- 4x R9700 on PCIe Gen5 x16 with **peer access working for every pair** (the Radiance preamble prints a
  `P2P access : ENABLED 0↔1 ✓ …` matrix at boot; `lspci -vv` should show a 32 GB BAR per card and
  `LnkSta: Speed 32GT/s, Width x16`). The push model is pointless without P2P.
- The Radiance stack as set up by `setup-paroquant.sh` (image `stilldeadcode/vllm-radiance:0.9.3`,
  vLLM 0.27.1 inside, Docker). Everything below assumes the repo clone at `~/vllm-mxfp4` and the
  libr4d cache at `~/.cache/radiance-libr4d/<key>`.
- A libr4d checkout with Radiance's `r4d_radiance_extras.patch` applied (that is what
  `setup-paroquant.sh` builds as key `b9e42ab-rx6`). The 4-rank kernel is templated on the 3-rank
  kernel that patch adds.

## 3. Build the library

Option A — use the full tree in this repo (`../../libr4d-ar4/`, rx6 + the 4-rank kernel):

```sh
cp -r inference-tuning/libr4d-ar4 ~/.cache/radiance-libr4d/b9e42ab-rx6-ar4
cd ~/.cache/radiance-libr4d/b9e42ab-rx6-ar4
make IMAGE=stilldeadcode/vllm-radiance:0.9.3          # ~3 min, all units, -> r4d.so
make verify IMAGE=stilldeadcode/vllm-radiance:0.9.3   # lists the kernels; expect ar_oneshot_4rank_exact
```

Option B — patch an existing rx6 tree:

```sh
cp -r ~/.cache/radiance-libr4d/b9e42ab-rx6 ~/.cache/radiance-libr4d/b9e42ab-rx6-ar4
cd ~/.cache/radiance-libr4d/b9e42ab-rx6-ar4 && rm -f *.o r4d.so
patch -p1 < /path/to/scripts/ar4/libr4d-ar4.patch      # adds the .hip, header decl, pybind, registry row, build unit
make IMAGE=stilldeadcode/vllm-radiance:0.9.3
```

Build inside the image: the extension must link against the same ROCm the container runs (7.14).
The host hipcc (ROCm 7.2.4 on monolith) crashes on this source anyway.

What the patch touches (all additive, nothing in the 2-rank path changes):

| file | change |
|---|---|
| `r4d_ar_oneshot_4rank_exact.hip` | the kernel + host entry `r4d_ar_oneshot_4rank_exact(...)` + `r4d_ar_4rank_max_blocks()` |
| `r4d.h` | declarations |
| `r4d_module.hip` | `m.def("ar_oneshot_4rank_exact", …)`, `AR4_MAX_BLOCKS` attr |
| `r4d_registry.hip` | a row with constraint `world_size == 4`, so `r4d.select("allreduce", world_size=4, exact=1, dtype="bf16")` returns it |
| `build.sh` | the translation unit and link object |

## 4. Teach Radiance's Python wrapper about 4 ranks

`radiance_allreduce.py` (in the image's site-packages) is written for two ranks in four places. The patch
script `patch_ar_4rank.py` (this directory; on monolith at `~/vllm-mxfp4/patch_ar_4rank.py`) rewrites them
at container start, in the same idempotent anchor/replace style as Radiance's own `patch_*.py`:

1. **Scratch/peer layout** follows the world size: `n_regions = ws - 1` receive regions × 2 slots ×
   `max_bytes`, one flag row per region, `peers` in ascending rank order, IPC handles opened for each.
2. **Message cap at ws ≥ 4**: `RADIANCE_AR4_MAX_KB` (default 768). The push moves 3x the message out of
   every card, so the one-shot wins below ~700 KB and loses above; prefill chunks (80 MiB at chunk 8192)
   stay on RCCL. Decode messages are `M × hidden × 2` bytes = 80 KB for an 8-wide verify.
3. **Fixed block count at ws ≥ 4**: `RADIANCE_AR4_NB` (default 8). The size heuristic is race-free too
   (see README) but one count keeps every block's sequence counter identical, which is the simplest
   invariant to reason about.
4. **Launch arm** for ws == 3 and ws == 4 (argument order: my scratch, peer scratches ascending, my flags,
   peer flags ascending, seq counters, slot words, rank, in, out, numel, dtype, stream, nblocks, nthreads,
   drain, acq).
5. **Compressed (wht6) path** disabled at ws ≠ 2 — there is no 3-/4-rank lossy kernel.
6. **Install gate**: `world_size in (2, 3, 4)`; which sizes actually get a kernel is still the library's
   answer via `r4d.select`.

It must run **after** `patch_ar_maxbytes.py` and `patch_ar_geometry.py`. It supersedes Radiance's
`patch_ar_3rank.py` (do not run both).

Diagnostics in the same patch, all off by default and safe to leave in:

| env | effect |
|---|---|
| `RADIANCE_AR4_SHADOW=1` | after every 4-rank call (eager serve only — graph replay never runs Python) recompute with RCCL on a clone, all-gather the outputs, log mismatches and cross-rank bit differences |
| `RADIANCE_AR4_MIN_KB=N` | messages below N KB go to RCCL (size-split experiments) |
| `RADIANCE_AR4_SKIP_FILES=<regex>` | call sites whose Python stack matches take RCCL; under CUDA graphs the choice is baked at capture, so `dflash\|spec_decode` routes the whole drafter to RCCL |

## 5. Launcher

`run_paroquant_ar4.sh` (this repo: `../launchers/`; monolith: `~/vllm-mxfp4/paroquant/`) is Radiance's
`run_paroquant.sh` plus:

- `python3 patch_ar_4rank.py` appended to the in-container patch chain (after `patch_ar_geometry.py`);
- env passthroughs: `RADIANCE_USE_R4D_AR` (1 = kernel, 0 = RCCL, for A/B), `RADIANCE_AR4_MAX_KB`,
  `RADIANCE_AR4_MIN_KB`, `RADIANCE_AR4_NB`, `RADIANCE_AR4_SHADOW`, `RADIANCE_AR4_SKIP_FILES`,
  `RADIANCE_DYNAMIC_DRAFT`, `VLLM_TORCH_PROFILER_DIR`, `EXTRA_VLLM_ARGS` (appended to `vllm serve`),
  and **`GPU_MAX_HW_QUEUES` (default 1)**.

Full diff vs upstream: `run_paroquant_ar4.diff`. Keep both files in sync when Radiance updates its launcher;
the diff is deliberately tiny.

Start a TP=4 server (what `tp4-ar4-launch.sh` does):

```sh
sudo systemctl stop vllm-qwen            # it holds GPUs 0,1
cd ~/vllm-mxfp4
RUNTIME=docker MODE=prod MODEL_DIR=Qwen3.8-27B-PARO-int5 SPEC=7 PORT=8090 TP=4 GPUS=0,1,2,3 \
  NAME=vllmtp4 CACHE=$HOME/.radiance-cache-paro-093-tp4 R4D_KEY=b9e42ab-rx6-ar4 \
  CHAT_TEMPLATE=$HOME/vllm-mxfp4/qwen-fixed-v22.3.jinja SERVED_NAMES="Qwen3.8 Qwen3.8-PARO Qwen3.8-27B" \
  ./paroquant/run_paroquant_ar4.sh
```

`R4D_KEY` selects the libr4d tree under `~/.cache/radiance-libr4d/`. First start on a new cache dir
compiles for ~11 min; warm restarts are ~7 min. The boot log must show, per worker:

```
[radiance] using patched r4d.so from ~/.cache/radiance-libr4d/b9e42ab-rx6-ar4
  OK    all-reduce 4-rank: install gate            (and the other patch_ar_4rank lines)
[radiance] custom all-reduce INSTALLED (rank=N max=768KB finegrained=True ... nb=8..8)
```

`custom AR disabled: no exact all-reduce kernel for world_size=4` means the r4d.so lacks the kernel
(wrong `R4D_KEY`). `AR_QUANT disabled: no lossy all-reduce kernel` at ws=4 is expected.

To make it the service: copy `deploy/vllm-qwen.service`, set `TP=4 GPUS=0,1,2,3 R4D_KEY=b9e42ab-rx6-ar4`,
point `ExecStart` at `run_paroquant_ar4.sh` and `ExecStop` at the container name, keep `PORT=8080` if
clients expect it. Nothing else on the box may hold the four cards.

## 6. `GPU_MAX_HW_QUEUES=1` — do not skip this

On gfx1201 a vLLM/Radiance process lands at spawn in one of two dispatch states. In the slow one every
small kernel in HIP-graph replay is followed by ~15 µs of idle instead of ~3 µs; with ~1,000 kernels per
decode step that is +7–10 ms per step, and the process stays there until restarted. Upstream:
ROCm/ROCm#6347 (bimodal R9700 decode, fixed at spawn), legacy-rocm-build#6685 (two or more streams with
pending `hipStreamWaitEvent` on other hardware queues add ~17 µs per kernel; gone with
`GPU_MAX_HW_QUEUES` ≤ 2). On monolith the one-shot configuration hit the slow state in 6 of 8 boots
(139/108/125 tok/s instead of 210/164/191); with `GPU_MAX_HW_QUEUES=1` every boot was fast and
concurrency 1–8 showed no cost. The launcher sets it by default; the plain RCCL configuration happened
to land fast every time but has no guarantee either.

How to tell which state a server is in without a benchmark: `RADIANCE_STEP_TRACE=50` prints per-step
timings; at single-stream decode with an 8-wide verify, `gpu_span` ≈ 18.5 ms is the fast state and
≈ 28 ms the slow one (RCCL: ≈ 22 ms). Nexus (single R9700) fixed the same thing with an idle-context
keeper (`../keeper/`); the flag is simpler.

## 7. Verify

Microbenchmark (no vLLM; a fresh container per run; needs the cards free):

```sh
PATCH=1 R4D=$HOME/.cache/radiance-libr4d/b9e42ab-rx6-ar4 ~/bench/ar-bench.sh 0,1,2,3
```

Prints per message size: RCCL eager/graph, one-shot eager/graph, and a check column — `ok` means the
sum matches an fp32 reference and all ranks hold identical bits (`bitexact` is vs RCCL, only reachable at
2 ranks where the summation order coincides). Expected at 4 ranks, HIP graph: 10 KB 6 µs vs 35, 80 KB
13 vs 38, 320 KB 30 vs 48, 640 KB 57 vs 62, 1.25 MB 104 vs 85 (hence the cap). Modes via env:
`AR_ROWS=…`, `AR_ITERS`, `AR_GEMM=1` (a layer-sized GEMM between calls), `AR_SKEW_CYCLES` (rank skew),
`AR_D2H_KB`, `AR_ALT=1` (alternate sizes/block counts under graph replay and check every output),
`AR_MIN_NB/AR_MAX_NB/AR_WPB/AR_NT/AR_DRAIN` (kernel geometry).

In-situ correctness (eager serve so Python sees every call):

```sh
RADIANCE_AR4_SHADOW=1 EXTRA_VLLM_ARGS="--enforce-eager" ./tp4-ar4-launch.sh   # then send requests
grep "AR4 SHADOW" <launch log> | tail -1      # want bad=0 rankdiff=0; maxrel ~1e-2 is bf16 rounding
```

Serving numbers: `URL=http://127.0.0.1:8090 MODEL=Qwen3.8 python3 bench_oai.py label 2` and
`python3 prefill_conc.py 8090 Qwen3.8`. `pair-hwq.sh` is the complete A/B (one-shot twice, RCCL once).

Profiling a real decode: launch with `PROFILE=1` (the launcher passes `--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=/cache/prof`), then `prof-step.sh LABEL` does start/stop around one
request and collects `rank*.pt.trace.json.gz`; `step_breakdown.py TRACE` prints per-step busy/idle and
per-category kernel time (all-reduce, target GEMM, GDN, drafter…), `trace_summary.py TRACE` the kernel
table and the idle gap around each all-reduce. This is how the dispatch state was caught.

## 8. Kernel design notes (for anyone changing it)

- **Push model.** Each block owns a contiguous chunk of 16-byte words of the message. It copies its chunk
  of the local input into all three peers' scratch (stores interleaved so one drain covers all), drains
  (`s_wait_storecnt 0` — correct for fine-grained/uncached memory, where posted PCIe writes to one peer
  stay ordered), releases one flag per peer with a system-scope release store, then thread 0 spins on its
  own three flags (system-scope acquire loads) until all reach this call's sequence number, and the
  block reduces its chunk: local input + three regions, in **canonical rank order**
  `((x0 + x1) + x2) + x3` in fp32, rounded to the payload dtype once. The rank only selects which operand
  is the local input; the order never changes, so all ranks hold the same bits (the replicated-state
  invariant Radiance's fused epilogues rely on). It is therefore *not* bit-identical to RCCL at 4 ranks
  (RCCL's ring sums in a different order); it is more exact, not less.
- **Scratch layout per rank**: 3 receive regions × 2 seq-parity slots × `max_bytes`; region index of
  sender i in receiver j is `i - (i > j)`, so regions hold the peers in ascending rank order. Flags:
  3 regions × `R4D_AR4_MAX_BLOCKS` (512) uints. IPC handles are exchanged once at install with
  `dist.all_gather_object`.
- **Graph safety**: the sequence number is a per-block device-resident counter the kernel increments,
  so replays stay correct and no host value is baked into a graph. Slot = `seq & 1`. A rank at seq s+2
  has passed the s+1 handshake with every peer, and a peer releases s+1 only after its kernel s finished
  reading slot s, so double-buffering suffices. With a size-dependent block count, blocks that take part
  in fewer calls have different counters; the heuristic (≥1,400 words per block) keeps any smaller
  message inside the blocks that took part in every call, so it is race-free, and the fixed count makes
  that trivially true.
- **Geometry**: 4..24 blocks × 1,024 threads in the stock heuristic; at 4 ranks 8 blocks fixed. The
  sweep (nb 4..24, nt 256..1024) is within noise at decode sizes; the cost is PCIe latency, not threads.
- **Not here**: a compressed (6-bit) 4-rank payload (the 2-rank wht6 path), and a fused
  all-reduce + RMSNorm + quant epilogue for 4 ranks (`radiance_arnq` gates itself to ws == 2). Both are
  the next steps if prefill or the fused path matter.

## 9. Adapting

- **Other TP widths**: TP=3 works through the same patch (Radiance's 3-rank kernel). TP=8 would need a
  7-peer kernel; the pattern generalises but 7x push traffic makes the one-shot a decode-only tool.
- **Other hidden sizes / verify widths**: the cap is in bytes; recheck the crossover with `ar-bench.sh`
  if messages change a lot (`AR_ROWS`).
- **Other RDNA4 cards**: build with `GFX_ARCH=gfx1200` etc. (`build.sh`). The dispatch-state issue is a
  gfx1201 report; test for it anyway (section 6).
- **If a boot hangs at capture or decodes garbage**: `RADIANCE_USE_R4D_AR=0` returns to RCCL with no
  other change — that is the A/B switch and the escape hatch.
