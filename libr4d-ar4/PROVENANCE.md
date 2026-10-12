# libr4d-ar4: provenance

This is the libr4d kernel library at upstream commit `b9e42ab` ("[FIX] correct the exponent clamp
rationale in gdn chunk_scan", https://codeberg.org/StillDeadcode/libr4d), with Radiance's
`r4d_radiance_extras.patch` (rx6: 2-rank `exact_nq`, 3-rank all-reduce, attention/GDN extras, from
https://github.com/jdrews/vllm-mxfp4) already applied, plus the work from this project:

- `r4d_ar_oneshot_4rank_exact.hip`, new: one-shot P2P push all-reduce for exactly four ranks.
- `r4d.h`, `r4d_module.hip`, `r4d_registry.hip`, `build.sh`: declaration, pybind binding, registry row
  (`world_size == 4`), build unit. Diff vs rx6: `../scripts/ar4/libr4d-ar4.patch`.

Build (inside the Radiance image; the host hipcc on monolith crashes):

    make IMAGE=stilldeadcode/vllm-radiance:0.9.3        # -> r4d.so for gfx1201
    make verify IMAGE=stilldeadcode/vllm-radiance:0.9.3

On monolith the built tree lives at `~/.cache/radiance-libr4d/b9e42ab-rx6-ar4` and is selected with
`R4D_KEY=b9e42ab-rx6-ar4`. Neither upstream repository ships a LICENSE file; this copy is kept in a
private repository for our own use. `.o`/`.so` build products are not committed.
