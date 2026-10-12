#!/usr/bin/env python3
"""Summarise a torch-profiler (kineto) trace: GPU kernel time by name, graph launches, and the
timeline around the all-reduce kernels. usage: trace_summary.py TRACE.json[.gz] [top]"""
import gzip, json, sys, collections, re
path = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
with open(path, "rb") as f:
    raw = f.read()
if raw[:2] == b"\x1f\x8b":
    raw = gzip.decompress(raw)
d = json.loads(raw.decode("utf-8", "replace"))
ev = d["traceEvents"] if isinstance(d, dict) else d
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
kern.sort(key=lambda e: e["ts"])
if not kern:
    cats = collections.Counter(e.get("cat") for e in ev); print("no kernel events; cats:", cats.most_common(10)); sys.exit()
t0, t1 = kern[0]["ts"], max(e["ts"] + e["dur"] for e in kern)
print(f"kernels: {len(kern)}  span: {(t1-t0)/1000:.1f} ms  busy: {sum(e['dur'] for e in kern)/1000:.1f} ms")
by = collections.defaultdict(lambda: [0, 0.0])
for e in kern:
    n = re.sub(r"<.*", "", e["name"])[:70]; by[n][0] += 1; by[n][1] += e["dur"]
print(f"{'count':>6} {'total ms':>9} {'mean us':>8}  name")
for n, (c, t) in sorted(by.items(), key=lambda kv: -kv[1][1])[:top]:
    print(f"{c:6d} {t/1000:9.2f} {t/c:8.1f}  {n}")
# all-reduce kernels: name match
ar = [e for e in kern if re.search(r"ar_oneshot|ncclDevKernel|ncclKernel|AllReduce|allreduce", e["name"], re.I)]
print(f"\nall-reduce-ish kernels: {len(ar)}  total {sum(e['dur'] for e in ar)/1000:.2f} ms  "
      f"mean {sum(e['dur'] for e in ar)/max(1,len(ar)):.1f} us")
if ar:
    # gap before and after each AR (idle between previous kernel end and AR start)
    idx = {id(e): i for i, e in enumerate(kern)}
    gaps_b, gaps_a = [], []
    for e in ar:
        i = idx[id(e)]
        if i > 0: gaps_b.append(e["ts"] - (kern[i-1]["ts"] + kern[i-1]["dur"]))
        if i + 1 < len(kern): gaps_a.append(kern[i+1]["ts"] - (e["ts"] + e["dur"]))
    import statistics as st
    print(f"idle gap before AR: mean {st.mean(gaps_b):.1f} us median {st.median(gaps_b):.1f}; after: mean {st.mean(gaps_a):.1f} median {st.median(gaps_a):.1f}")
    names = collections.Counter(re.sub(r"<.*", "", e["name"])[:60] for e in ar); print(names.most_common(5))
# total idle on the GPU timeline
idle = 0.0; last = kern[0]["ts"] + kern[0]["dur"]
for e in kern[1:]:
    if e["ts"] > last: idle += e["ts"] - last
    last = max(last, e["ts"] + e["dur"])
print(f"GPU idle within span: {idle/1000:.1f} ms ({100*idle/(t1-t0):.0f}%)")
cpu = collections.Counter(e["name"][:50] for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver", "hip_runtime"))
print("\nruntime API calls:", cpu.most_common(8))
