# Upstream submissions (2026-10-12)


- **Radiance PR:** https://github.com/jdrews/vllm-mxfp4/pull/2, branch `tp4-oneshot-allreduce` on the
  fork https://github.com/brainchillz/vllm-mxfp4. Contents here: `r4d_radiance_extras_rx11.patch` (rx6 +
  the 4-rank kernel, applies to upstream libr4d `b9e42ab`), the lean `patch_ar_4rank.py` (shadow check and
  fixed block count kept, the hunt-only `MIN_KB` / `SKIP_FILES` diagnostics dropped), `launchers-and-docs.diff`
  (serve-mxfp4.sh, run_paroquant.sh, PERFORMANCE.md row, README TP=4 section), and `pr-body.md`.
- **ROCm/legacy-rocm-build#6347** (bimodal R9700 decode): comment with the TP=4 data, profiler
  breakdown and `GPU_MAX_HW_QUEUES=1` results:
  https://github.com/ROCm/legacy-rocm-build/issues/6347#issuecomment-6119110147 (`c6347.md`).
- **ROCm/legacy-rocm-build#6685** (pending-wait dispatch penalty): comment with the in-serve numbers and
  the author's reproducer run on this box: `c6685.md`; the reproducer source (`gfx1201-pending-wait-repro.hip`, from that issue) and our run are in the comment.
