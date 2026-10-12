# libr4d

R4D is a library of specialized HIP kernels for **gfx1201 (RDNA4, Radeon AI PRO R9700)**.

It builds to a single importable extension, `r4d.so`.

## Naming

An entry point is

```
<family>_<op>_<the geometry it is compiled for>
```

and the geometry is not decoration. These are specialised kernels: every dimension in the name is a
compile-time constant, and the entry point **rejects a mismatch rather than running**.
`attn_decode_h256_gqa6_fp8kv` serves head_dim 256 with 6 queries per KV head and an fp8-e4m3 paged
cache, and nothing else. A model with a different head size needs a new instantiation, which gets
its own name beside this one -- no existing name has to change to make room for it, and no caller
can pick up a kernel that does not fit by accident.

Dimensions that are the same across the whole library discriminate nothing between entry points and
so stay out of the names: the paged block size (16), the query dtype (bf16 everywhere) and the
target architecture. They are reported by the `ATTN_*` / `GDN_*` / `AR_*` / `GEMM_*` module
constants and by `kernels()`.

| entry point | what it is |
| --- | --- |
| `attn_prefill_h256_gqa6_fp8kv` / `..._bf16kv` | paged causal attention, query-tiled; head_dim 256, 6 queries per KV head, fp8-e4m3 or bf16 KV cache |
| `attn_decode_h256_gqa6_fp8kv` / `..._bf16kv` | the same geometry, split-KV, up to 64 query rows |
| `attn_vit_h72_bf16` | dense non-causal vision-encoder attention, head_dim 72 native, varlen over images |
| `gdn_conv_prep_w4_h128_bf16` | the gdn prefill preamble in one kernel: causal conv (width 4, silu, with its state cache), q/k/v split, qk l2norm, gating and the per-chunk gate cumsum |
| `gdn_kkt_solve_k128_c64_bf16` | `A = (I + strict_lower(diag(beta) K K^T e^dg))^-1` per chunk of 64 -- the gram and its triangular inverse, without the fp32 gram ever reaching memory |
| `gdn_chunk_scan_k128_v128_c64_bf16` | gated-delta-net chunked scan -- WY recompute, state recurrence and output in one kernel; head_k 128, head_v 128, chunk 64 |
| `gdn_conv_update_w4_h128_bf16` | the same convolution for a decode step: a rolling speculative window over the conv state cache, writing q/k/v directly |
| `gdn_recurrent_update_k128_v128_bf16_fp32state` | the decode-time recurrent delta-rule update against the paged fp32 state cache, gating and qk l2norm included |
| `gdn_gated_rmsnorm_h128_bf16` | gated rms norm over a 128-channel row, `out = rms(x) . w . act(z)` |
| `ar_oneshot_2rank_exact` | one-shot push all-reduce over P2P, exactly 2 ranks, exact bf16/fp16/fp32 |
| `ar_oneshot_2rank_wht6` | the same handshake with a Walsh-Hadamard-rotated 6-bit wire payload (lossy) |
| `gemm_bf16_nt_m16` | skinny bf16 GEMM, `C[M,N] = A[M,K] @ W[N,K]^T`, M ≤ 16, split-K |

## Asking the library instead of remembering

A geometry used to be written down three times: in the entry point's name, in the check the entry
point rejects a mismatch on, and again in whatever caller decided it was allowed to make the call.
The third copy is the one that drifts -- it lives in another repository and nothing ties it back to
the kernel. So the constraints are data, and a caller asks:

```python
>>> import r4d, torch
>>> r4d.select("attn_prefill_paged", head_dim=256, gqa=6, block_size=16, causal=1,
...            q_dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn)
'attn_prefill_h256_gqa6_fp8kv'
>>> r4d.select("attn_prefill_paged", head_dim=512, gqa=1, block_size=16, causal=1,
...            q_dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn)      # Gemma-4's global layers
None
```

`None` is a real answer, and the useful one: **this build has no kernel for that geometry, so use
whatever you would have used without R4D.** Every model the library was not compiled for gets it.

What `select()` settles is whether a kernel *exists* for a shape. It is a necessary condition, not
a sufficient one -- contiguity, strides, alignment and buffer sizes are properties of the tensors at
a call rather than of a model, and stay in the caller's own check and in the entry point, which
still rejects what it cannot run.

* `r4d.ops()` -- every op, mapped to the parameters `select()` will ask for
* `r4d.explain(op, **geometry)` -- every candidate, and which constraint failed
* `r4d.constraints(name)` -- one kernel's geometry as `{key, op, value}` rows
* `r4d.kernels()` -- the whole table: name, family, op, what it computes, the geometry in prose and
  as constraints, and the operand dtypes
* `r4d.selections()` -- every distinct question `select()` has been asked in this process, in the
  order first asked, as `{op, query, kernel, reason, count}` rows
* the `ATTN_*` / `GDN_*` / `AR_*` / `GEMM_*` module constants -- the same numbers as scalars

```python
>>> r4d.explain("attn_prefill_paged", head_dim=256, gqa=8, block_size=16, causal=1,
...             q_dtype="bf16", kv_dtype="fp8_e4m3")
[{'name': 'attn_prefill_h256_gqa6_fp8kv',  'ok': False, 'reason': 'gqa=8, needs == 6'},
 {'name': 'attn_prefill_h256_gqa6_bf16kv', 'ok': False, 'reason': 'gqa=8, needs == 6'}]
```

`selections()` exists because callers ask their questions at import time, scattered across modules,
long before there is a log anyone is reading. The library keeps the questions so a host can report
what it resolved to once it is up -- vllm-radiance prints exactly this table after the worker
finishes warming up. Repeats are folded with a count, since one attention layer asks per
instantiation; keyword order does not split a row, but a different value does, and a `torch.dtype`
folds into the same row as its short name. A query that resolved to nothing is kept too, with the
constraint that refused it, because that is the record of why a fallback is running.

A dtype may be a `torch.dtype`, its `str()`, or the short name (`bf16`, `fp16`, `fp32`,
`fp8_e4m3`). Adding a kernel means adding a row to `r4d_registry.hip`, which is what makes it
visible to all of the above.

The kernels are plain HIP behind the C ABI in [`r4d.h`](r4d.h); `r4d_module.hip` is the only file
that knows about Python, so the same sources compile into a non-Python consumer unchanged.

## Building

```sh
./build.sh                      # -> r4d.so for gfx1201
GFX_ARCH=gfx1200 ./build.sh
make IMAGE=some/rocm-image      # or build it inside a container
make verify IMAGE=some/rocm-image
```

It needs `hipcc` and the pybind11 headers, and it must be compiled with the same ROCm the process
that imports it links against.

Translation units are compiled separately and linked, so per-kernel target features stay scoped:
`-mcumode` applies to `r4d_gdn_chunk_scan_k128_v128_c64_bf16.o` alone, and `-ffp-contract=off` is global because the rotated
6-bit all-reduce requires it (the two ranks fuse different products otherwise and diverge by ~1 ULP).

## Adding a kernel

1. Write the `.hip`; name the entry point for what it computes and the geometry it is compiled for.
2. Declare it in `r4d.h`, and add a `_dims()` accessor if it has compiled-in geometry a caller
   should be able to read.
3. Name the file after the entry point, geometry included; only shared machinery (`r4d_common.h`,
   `r4d_dt16.h`, `r4d_gdn_wmma.h`) is named for what it is, because it provides no kernel.
4. Add a row to `r4d_registry.hip` -- that table is what `kernels()`, `ops()` and `select()` report,
   so a kernel that is not in it is invisible to every caller that asks rather than remembers.
   Give it the `op` of the operation it implements, which it may share with other rows, and write
   its geometry as constraints as well as prose.
5. Bind it in `r4d_module.hip` under the same name minus the `r4d_` prefix.
6. Add the translation unit to `UNITS` in `build.sh`.

## Notes for anyone reading the kernels

Three things about this architecture cost the most time to find, and they are assumed throughout:

* The wave32 WMMA fragment layout is `idx = lane % 16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`. Both
  A and B want K-contiguous rows, so swapping the operands of a WMMA transposes the result for free.
* gfx1201 has no `v_cvt_pk_bf16_f32`, no direct-to-LDS, and no `ds_read_b64_tr_b16`. Packing bf16 is
  software, and the cheapest form is two adds plus a `v_perm_b32`.
* LLVM single-buffers LDS and drains before every WMMA unless you tell it not to.
  `__builtin_amdgcn_sched_group_barrier` is the fix, and the granularity of the groups matters far
  more than their contents -- coarser is better, up to the point where the pattern asks for more
  outstanding loads than the hardware can hold.
