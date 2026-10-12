#!/usr/bin/env python3
"""All-reduce latency microbenchmark: RCCL vs the libr4d one-shot P2P kernel (radiance_allreduce).

Run inside the vllm-radiance image with torchrun --nproc_per_node=N. Measures per-call latency of
back-to-back all-reduces at decode-size messages (bf16, hidden=5120 rows) and a few larger ones.
At world_size != 2/3 the r4d wrapper disables itself and only RCCL is reported.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

HIDDEN = 5120
ROWS = [int(r) for r in os.environ.get("AR_ROWS", "1,8,16,32,64,128,256,512,1024,4096").split(",")]
WARMUP = 30
ITERS = int(os.environ.get("AR_ITERS", "300"))


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def time_fn(fn, iters):
    # back-to-back launches on the current stream, like consecutive layers at decode
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us per call


def time_graph(fn, iters):
    # same calls captured once in a HIP graph and replayed: removes host launch overhead
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=s):
            for _ in range(iters):
                fn()
    except Exception as e:  # noqa: BLE001
        return None, repr(e)
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters, None


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    cpu_group = dist.new_group(backend="gloo")
    dist.barrier()

    log(rank, f"world_size={world} device={torch.cuda.get_device_name(local)} "
              f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}")

    comm = None
    try:
        import radiance_allreduce as ra
        comm = ra.RadianceAllreduce(cpu_group, dev)
        if comm.disabled:
            comm = None
        else:
            for env, attr in (("AR_MIN_NB", "min_nb"), ("AR_MAX_NB", "max_nb"),
                              ("AR_WPB", "words_per_block"), ("AR_NT", "nt"), ("AR_DRAIN", "drain")):
                if os.environ.get(env):
                    setattr(comm, attr, int(os.environ[env]))
            log(rank, f"r4d geometry: min_nb={comm.min_nb} max_nb={comm.max_nb} "
                      f"wpb={comm.words_per_block} nt={comm.nt} drain={comm.drain}")
    except Exception as e:  # noqa: BLE001
        log(rank, f"radiance_allreduce unavailable: {e!r}")
    log(rank, f"r4d one-shot kernel: {'INSTALLED' if comm else 'not available at this world size'}")

    # ---- race test: alternate message sizes (different block counts) back to back under HIP graph
    # replay with rank-0 skew, no host sync; then check every output against an exact reference.
    ALT = int(os.environ.get("AR_ALT", "0"))
    if ALT and comm is not None:
        sizes = [int(r) for r in os.environ.get("AR_ALT_ROWS", "8,64").split(",")]
        skew = int(os.environ.get("AR_SKEW_CYCLES", "20000"))
        iters = int(os.environ.get("AR_ALT_ITERS", "200"))
        xs = [torch.randn(r, HIDDEN, dtype=torch.bfloat16, device=dev) * (rank + 1) for r in sizes]
        refs = []
        for x in xs:
            g = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(g, x)
            refs.append(sum(t.float() for t in g).to(torch.bfloat16))
        outs = [torch.empty_like(x) for x in xs]
        log(rank, f"race test: rows={sizes} nb={[comm._nblocks(x.numel()*2//16) for x in xs]} "
                  f"skew_cycles(rank0)={skew} iters={iters} graph replay, min_nb={comm.min_nb} max_nb={comm.max_nb}")
        def seq():
            for i, x in enumerate(xs):
                if skew and rank == (i % world):
                    torch.cuda._sleep(skew)
                outs[i].copy_(comm.custom_all_reduce(x))
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            for _ in range(3):
                seq()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            for _ in range(iters):
                seq()
        bad = 0; worst = 0.0; rankdiff = 0
        for rep in range(int(os.environ.get("AR_ALT_REPS", "20"))):
            g.replay()
            torch.cuda.synchronize()
            for i in range(len(xs)):
                d = (outs[i].float() - refs[i].float()).abs().max().item()
                worst = max(worst, d)
                if d > 1e-1 * refs[i].float().abs().max().item():
                    bad += 1
                gat = [torch.empty_like(outs[i]) for _ in range(world)]
                dist.all_gather(gat, outs[i])
                if not all(torch.equal(gat[0], o) for o in gat[1:]):
                    rankdiff += 1
        log(rank, f"race test result: checks={len(xs)*20} wrong={bad} rankdiff={rankdiff} worst_abs_err={worst:.3f}")
        dist.barrier(); dist.destroy_process_group(); return

    # ---- "in situ" mode: a layer-sized GEMM between consecutive all-reduces (cold caches, natural
    # skew), optional extra skew on rank 0 (torch.cuda._sleep cycles), optional D2H traffic.
    # Reports (gemm + AR) - gemm as the per-call AR cost, HIP-graph replay.
    GEMM = int(os.environ.get("AR_GEMM", "0"))
    SKEW = int(os.environ.get("AR_SKEW_CYCLES", "0"))
    D2H = int(os.environ.get("AR_D2H_KB", "0"))
    if GEMM or SKEW or D2H:
        g = torch.Generator(device=dev); g.manual_seed(1234)   # same W on every rank
        W = torch.randn(5120, 5120, dtype=torch.bfloat16, device=dev, generator=g)
        hbuf = torch.empty(max(D2H, 1) * 1024, dtype=torch.uint8, device="cpu", pin_memory=True)
        dbuf = torch.zeros(max(D2H, 1) * 1024, dtype=torch.uint8, device=dev)
        log(rank, f"in-situ mode: gemm={GEMM} skew_cycles(rank0)={SKEW} d2h_kb={D2H}")
        hdr2 = f"{'rows':>5} {'bytes':>9} | {'filler':>8} | {'rccl+f':>8} {'rccl AR':>8} | {'r4d+f':>8} {'r4d AR':>8} | {'check':>8}"
        log(rank, hdr2); log(rank, "-" * len(hdr2))
        for rows in ROWS:
            x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=dev) * (rank + 1)
            nbytes = x.numel() * 2
            def filler():
                if GEMM:
                    y = x @ W
                    x.copy_(y * 1e-3 + x)   # keep values bounded, data-dependent
                if SKEW and rank == 0:
                    torch.cuda._sleep(SKEW)
                if D2H:
                    hbuf.copy_(dbuf, non_blocking=True)
            def rccl_f():
                filler(); dist.all_reduce(x)
            def r4d_f():
                filler(); out = comm.custom_all_reduce(x); x.copy_(out)
            n = 60
            t_f, _ = time_graph(filler, n)
            t_rccl, e1 = time_graph(rccl_f, n)
            t_r4d = None; chk = "-"
            if comm is not None and comm.should_custom_ar(x):
                t_r4d, e2 = time_graph(r4d_f, n)
                outs = [torch.empty_like(x) for _ in range(world)]
                dist.all_gather(outs, x)
                chk = "ok" if all(torch.equal(outs[0], o) for o in outs[1:]) else "RANKDIFF"
            def f(v):
                return f"{v:8.1f}" if isinstance(v, float) else f"{'n/a':>8}"
            def d(v):
                return f"{v - t_f:8.1f}" if isinstance(v, float) and isinstance(t_f, float) else f"{'n/a':>8}"
            log(rank, f"{rows:5d} {nbytes:9d} | {f(t_f)} | {f(t_rccl)} {d(t_rccl)} | {f(t_r4d)} {d(t_r4d)} | {chk:>8}")
            if e1 and rows == ROWS[0]:
                log(rank, f"  (graph capture failed: {str(e1)[:160]})")
            dist.barrier()
        dist.barrier(); dist.destroy_process_group(); return

    hdr = f"{'rows':>5} {'bytes':>9} | {'rccl eager':>10} {'rccl graph':>10} | {'r4d eager':>10} {'r4d graph':>10} | {'check':>8}"
    log(rank, hdr)
    log(rank, "-" * len(hdr))
    for rows in ROWS:
        x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=dev) * (rank + 1)
        nbytes = x.numel() * 2

        def rccl():
            y = x.clone()
            dist.all_reduce(y)
            return y

        t_rccl = time_fn(rccl, ITERS)
        g_rccl, g_err = time_graph(rccl, min(ITERS, 100))

        t_r4d = g_r4d = None
        exact = "-"
        if comm is not None and comm.should_custom_ar(x):
            def r4d():
                return comm.custom_all_reduce(x)
            ref = rccl()
            out = r4d()
            torch.cuda.synchronize()
            # exact fp32 reference of the sum, so a wrong operand/ordering shows as MISMATCH
            xs = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(xs, x)
            exact_ref = sum(t.float() for t in xs)
            ok_math = torch.allclose(exact_ref, out.float(), rtol=1e-2, atol=1e-2)
            # every rank must hold the same bits (replicated-state invariant)
            outs = [torch.empty_like(out) for _ in range(world)]
            dist.all_gather(outs, out)
            ranks_same = all(torch.equal(outs[0], o) for o in outs[1:])
            same = torch.equal(ref, out)
            if not ok_math:
                exact = "MISMATCH"
            elif not ranks_same:
                exact = "RANKDIFF"
            else:
                exact = "bitexact" if same else "ok"
            t_r4d = time_fn(r4d, ITERS)
            g_r4d, _ = time_graph(r4d, min(ITERS, 100))

        def f(v):
            return f"{v:10.1f}" if isinstance(v, float) else f"{'n/a':>10}"
        log(rank, f"{rows:5d} {nbytes:9d} | {f(t_rccl)} {f(g_rccl)} | {f(t_r4d)} {f(g_r4d)} | {exact:>8}")
        if g_err and rows == ROWS[0]:
            log(rank, f"  (graph capture of RCCL failed: {g_err[:120]})")
        dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
