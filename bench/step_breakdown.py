#!/usr/bin/env python3
"""Per-decode-step breakdown of a vLLM kineto trace: busy/idle, all-reduce, drafter, target GEMM,
graph launches. usage: step_breakdown.py TRACE"""
import gzip, json, sys, re, statistics as st, collections
raw = open(sys.argv[1], "rb").read()
if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
ev = json.loads(raw.decode("utf-8", "replace"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
kern = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
ann = sorted([e for e in X if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context")], key=lambda e: e["ts"])
gen = [a for a in ann if "generation_1(8)" in a["name"] or re.search(r"generation_\d+\(\d+\)", a["name"]) and "generation_0" not in a["name"]]
print(f"gpu annotations: {len(ann)}  generation steps: {len(gen)}  kernels: {len(kern)}")
if gen:
    print(f"first->last step wall: {(gen[-1]['ts']+gen[-1]['dur']-gen[0]['ts'])/1000:.1f} ms over {len(gen)} steps = {(gen[-1]['ts']+gen[-1]['dur']-gen[0]['ts'])/1000/len(gen):.2f} ms/step incl. gaps")
cats = {
  "all_reduce": r"ar_oneshot|ncclDevKernel",
  "target_gemm": r"pq_int4_fp8_gemm|Cijk_",
  "gdn": r"r4d_gdn|gdn",
  "attn": r"r4d_attn|unified_attention|reshape_and_cache",
  "drafter": r"_draft_head|a8w8_blockscale|_rerank|_selector|preshuffle",
  "norm_rot": r"pq_add_rms_rot|pq_ew_rot|rms",
  "sampler": r"topk|bitonic|gatherTopK|Sort",
}
def cat_of(n):
    for c, p in cats.items():
        if re.search(p, n): return c
    return "other"
cpu = [e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver", "hip_runtime", "runtime")]
graph_launch = [e for e in cpu if "GraphLaunch" in e["name"]]
kl = [e for e in cpu if re.search(r"LaunchKernel", e["name"])]
print(f"CPU launches total: graph {len(graph_launch)}  kernel {len(kl)}")
rows = []
ki = 0
for a in gen:
    t0, t1 = a["ts"], a["ts"] + a["dur"]
    ks = [e for e in kern if e["ts"] >= t0 and e["ts"] < t1]
    busy = 0.0; last = t0; per = collections.Counter(); cnt = collections.Counter()
    for e in ks:
        s, d = e["ts"], e["dur"]; en = s + d
        if en > last: busy += en - max(s, last); last = en
        c = cat_of(e["name"]); per[c] += d; cnt[c] += 1
    gl = sum(1 for e in graph_launch if t0 <= e["ts"] < t1)
    rows.append(dict(dur=a["dur"], busy=busy, idle=a["dur"] - busy, n=len(ks), gl=gl, **{f"t_{c}": per[c] for c in list(cats) + ["other"]}, **{f"n_{c}": cnt[c] for c in ["all_reduce", "drafter"]}))
def med(k): return st.median(r[k] for r in rows) / 1000 if rows else 0
def medn(k): return st.median(r[k] for r in rows) if rows else 0
print(f"\nper step (median over {len(rows)} steps), ms:")
print(f"  step dur {med('dur'):6.2f}  busy {med('busy'):6.2f}  idle {med('idle'):6.2f}   kernels/step {medn('n'):.0f}  graph launches/step {medn('gl'):.0f}")
for c in list(cats) + ["other"]:
    print(f"  {c:12s} {med('t_'+c):6.2f}" + (f"   (n={medn('n_'+c):.0f})" if 'n_'+c in rows[0] else ""))
