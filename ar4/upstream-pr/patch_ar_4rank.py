#!/usr/bin/env python3
"""Teach radiance_allreduce.py the three- and four-rank one-shot kernels (r4d ar_oneshot_3rank_exact,
ar_oneshot_4rank_exact).

The shipped module is written for exactly two ranks in four places: it opens ONE peer's IPC
handles (`peer = 1 - self.rank`), sizes scratch and flags for one sender, launches the 2-rank
entry point's argument list, and install_custom_ar attaches only when the TP group is 2 wide.
At TP=3 and TP=4 every one of those silently hands the reduction to RCCL. This patch makes the
peer list, the scratch layout and the launch follow the world size, and lets a 3- or 4-wide group
attach. It is patch_ar_3rank.py extended by one rank; the two are alternatives, not a chain.

Which kernel serves a group is still the library's answer (`r4d.select("allreduce",
world_size=...)`), so an r4d.so without the 4-rank unit leaves TP=4 on RCCL exactly as today, with
the same "no exact all-reduce kernel for world_size=4" line in the log. TP=2 is untouched: the
peer list has one entry, the scratch is the same 2 slots, and the launch is the same call.

Scratch at ws=4 is 3 receive regions x 2 slots x max_bytes, and every call pushes the message to
three peers, so the one-shot only wins at decode sizes (measured 2026-10-11: RCCL ~40 us per call
at 10-640 KB vs the 2-rank one-shot at 5-7 us; a 4-rank push of an 80 MiB prefill message would
move 240 MiB per card). RADIANCE_AR4_MAX_KB (default 768) caps the kernel at ws=4; prefill
rides RCCL. The compressed (wht6) path has no 3-/4-rank kernel and disables itself.

RADIANCE_AR4_NB (default 8) gives every call one block count, so every block's sequence counter
advances on every call and the two scratch slots stay race-free by construction. RADIANCE_AR4_SHADOW=1
(eager serve only) recomputes every 4-rank call with RCCL and all-gathers the outputs: a correctness
gate for a new box, not a serving mode.

Runs after patch_ar_maxbytes.py and patch_ar_geometry.py (independent anchors); supersedes
patch_ar_3rank.py (do not run both). Idempotent; fatal-loud on drift."""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
AR = SP / "radiance_allreduce.py"

ALLOC_ANCHOR = (
    "            # double-buffered scratch (2 slots) + per-block flags, both IPC-shared\n"
    "            self._scratch, sc_h, fine_used = self._alloc(2 * self.max_bytes, fine)\n"
    "            self._flags, fl_h, _ = self._alloc(maxb * 4, fine)\n"
    "            sc_handles = [None] * self.world_size\n"
    "            fl_handles = [None] * self.world_size\n"
    "            dist.all_gather_object(sc_handles, sc_h, group=group)\n"
    "            dist.all_gather_object(fl_handles, fl_h, group=group)\n"
    "            peer = 1 - self.rank\n"
    "            self._peer_scratch = ext.ar_ipc_open(sc_handles[peer])\n"
    "            self._peer_flags = ext.ar_ipc_open(fl_handles[peer])\n"
)
ALLOC_NEW = (
    "            # double-buffered scratch (2 slots) + per-block flags, both IPC-shared. One receive\n"
    "            # region per SENDER (patch_ar_4rank.py): ws=2 keeps its single region, ws=3 holds\n"
    "            # two and ws=4 three, each 2 slots x max_bytes, with one flag row of maxb each.\n"
    "            self.n_regions = self.world_size - 1\n"
    "            self._scratch, sc_h, fine_used = self._alloc(self.n_regions * 2 * self.max_bytes, fine)\n"
    "            self._flags, fl_h, _ = self._alloc(self.n_regions * maxb * 4, fine)\n"
    "            sc_handles = [None] * self.world_size\n"
    "            fl_handles = [None] * self.world_size\n"
    "            dist.all_gather_object(sc_handles, sc_h, group=group)\n"
    "            dist.all_gather_object(fl_handles, fl_h, group=group)\n"
    "            # peers in ascending rank order: the 3-/4-rank kernels derive each region from that\n"
    "            self.peers = [r for r in range(self.world_size) if r != self.rank]\n"
    "            self._peer_scratch_all = [ext.ar_ipc_open(sc_handles[p]) for p in self.peers]\n"
    "            self._peer_flags_all = [ext.ar_ipc_open(fl_handles[p]) for p in self.peers]\n"
    "            # the 2-rank names, used by the 2-rank launch below and by radiance_arnq\n"
    "            self._peer_scratch = self._peer_scratch_all[0]\n"
    "            self._peer_flags = self._peer_flags_all[0]\n"
)

# ws=4 message cap: the push moves 3x the message out of each card, so the one-shot only pays at
# decode sizes. Applied before the 16B rounding so slot16 follows it.
CAP_ANCHOR = (
    "        self.max_bytes = _MAX_BYTES\n"
    "        self.max_bytes = (self.max_bytes // 16) * 16     # 16B (uint4) alignment\n"
)
CAP_NEW = (
    "        self.max_bytes = _MAX_BYTES\n"
    "        if self.world_size >= 4:   # patch_ar_4rank.py: decode-size messages only at ws=4\n"
    "            _cap = int(os.environ.get(\"RADIANCE_AR4_MAX_KB\") or 768) * 1024\n"
    "            self.max_bytes = min(self.max_bytes, _cap)\n"
    "        self.max_bytes = (self.max_bytes // 16) * 16     # 16B (uint4) alignment\n"
)

LAUNCH_ANCHOR = (
    "        else:\n"
    "            self._ar_exact(\n"
    "                self._peer_scratch, self._scratch, self._peer_flags, self._flags,\n"
    "                self._seq.data_ptr(), self.slot16,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "        return out\n"
)
LAUNCH_NEW = (
    "        elif self.world_size == 3:\n"
    "            # patch_ar_4rank.py: (my scratch, lower peer, higher peer, my flags, lower peer\n"
    "            # flags, higher peer flags, seq, slot words, rank, ...). Same block heuristic.\n"
    "            self._ar_exact(\n"
    "                self._scratch, self._peer_scratch_all[0], self._peer_scratch_all[1],\n"
    "                self._flags, self._peer_flags_all[0], self._peer_flags_all[1],\n"
    "                self._seq.data_ptr(), self.slot16, self.rank,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "        elif self.world_size == 4:\n"
    "            # patch_ar_4rank.py: (my scratch, 3 peer scratches ascending, my flags, 3 peer\n"
    "            # flags ascending, seq, slot words, rank, ...). Same block heuristic.\n"
    "            self._ar_exact(\n"
    "                self._scratch, self._peer_scratch_all[0], self._peer_scratch_all[1],\n"
    "                self._peer_scratch_all[2],\n"
    "                self._flags, self._peer_flags_all[0], self._peer_flags_all[1],\n"
    "                self._peer_flags_all[2],\n"
    "                self._seq.data_ptr(), self.slot16, self.rank,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "            if _SHADOW and not torch.cuda.is_current_stream_capturing():\n"
    "                self._shadow_check(inp, out)\n"
    "        else:\n"
    "            self._ar_exact(\n"
    "                self._peer_scratch, self._scratch, self._peer_flags, self._flags,\n"
    "                self._seq.data_ptr(), self.slot16,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "        return out\n"
)

QUANT_ANCHOR = (
    "        if not self.ar_quant or self._qext is None:\n"
    "            return False\n"
    "        if inp.dtype not in (torch.bfloat16, torch.float16):\n"
)
QUANT_NEW = (
    "        if not self.ar_quant or self._qext is None:\n"
    "            return False\n"
    "        if self.world_size != 2:      # patch_ar_4rank.py: no compressed 3-/4-rank kernel\n"
    "            return False\n"
    "        if inp.dtype not in (torch.bfloat16, torch.float16):\n"
)

SHADOW_ANCHOR = (
    "    @contextmanager\n"
    "    def capture(self):\n"
)
SHADOW_NEW = (
    "    # patch_ar_4rank.py diagnostic (RADIANCE_AR4_SHADOW=1, eager serve only): after every 4-rank\n"
    "    # one-shot, recompute the sum with RCCL on a clone and check (a) the kernel matches it within\n"
    "    # bf16 tolerance and (b) all ranks hold identical bits. Logs a running tally on rank 0.\n"
    "    _shadow_stats = {\"n\": 0, \"bad\": 0, \"rankdiff\": 0, \"maxrel\": 0.0, \"sizes\": {}}\n"
    "    def _shadow_check(self, inp, out):\n"
    "        st = self._shadow_stats\n"
    "        ref = inp.clone()\n"
    "        dist.all_reduce(ref)\n"
    "        gathered = [torch.empty_like(out) for _ in range(self.world_size)]\n"
    "        dist.all_gather(gathered, out)\n"
    "        torch.cuda.synchronize()\n"
    "        same = all(torch.equal(gathered[0], g) for g in gathered[1:])\n"
    "        diff = (ref.float() - out.float()).abs().max().item()\n"
    "        scale = ref.float().abs().max().item() or 1.0\n"
    "        rel = diff / scale\n"
    "        st[\"n\"] += 1\n"
    "        st[\"maxrel\"] = max(st[\"maxrel\"], rel)\n"
    "        k = inp.numel() * inp.element_size()\n"
    "        st[\"sizes\"][k] = st[\"sizes\"].get(k, 0) + 1\n"
    "        if rel > 2e-2:\n"
    "            st[\"bad\"] += 1\n"
    "        if not same:\n"
    "            st[\"rankdiff\"] += 1\n"
    "        if self.rank == 0 and (st[\"n\"] <= 5 or st[\"n\"] % 500 == 0 or rel > 2e-2 or not same):\n"
    "            _log(f\"AR4 SHADOW n={st['n']} bad={st['bad']} rankdiff={st['rankdiff']} this: bytes={k} \"\n"
    "                 f\"rel={rel:.3e} same_ranks={same} | maxrel={st['maxrel']:.3e} sizes={dict(sorted(st['sizes'].items()))}\")\n"
    "\n"
    "    @contextmanager\n"
    "    def capture(self):\n"
)
_SHADOW_ANCHOR = (
    "_QUANT_MIN_BYTES = int(os.environ.get(\"RADIANCE_AR_QUANT_MIN_KB\") or 128) * 1024\n"
)
_SHADOW_NEW = (
    "_QUANT_MIN_BYTES = int(os.environ.get(\"RADIANCE_AR_QUANT_MIN_KB\") or 128) * 1024\n"
    "_SHADOW = os.environ.get(\"RADIANCE_AR4_SHADOW\", \"0\") == \"1\"   # patch_ar_4rank.py diagnostic\n"
)

NB_ANCHOR = (
    "        self.max_nb = min(24, int(ext.AR_MAX_BLOCKS))\n"
    "        self.words_per_block = 1400   # 16B words/block target for the block-count heuristic\n"
)
NB_NEW = (
    "        self.max_nb = min(24, int(ext.AR_MAX_BLOCKS))\n"
    "        self.words_per_block = 1400   # 16B words/block target for the block-count heuristic\n"
    "        if self.world_size >= 4:\n"
    "            # patch_ar_4rank.py: RADIANCE_AR4_NB (default 8) = ONE block count for every call. The\n"
    "            # per-block seq counters only keep the two scratch slots race-free if every block takes\n"
    "            # part in every call; a size-dependent block count lets a fast rank's wide call overwrite\n"
    "            # words a slow peer is still reducing from the previous narrow call. 0 = heuristic.\n"
    "            _nb = int(os.environ.get(\"RADIANCE_AR4_NB\") or 8)\n"
    "            if _nb > 0:\n"
    "                self.min_nb = self.max_nb = min(_nb, int(ext.AR_MAX_BLOCKS))\n"
)

GATE_ANCHOR = (
    "            if \"tp\" in getattr(self, \"unique_name\", \"\") and getattr(self, \"world_size\", 1) == 2:\n"
)
GATE_NEW = (
    "            # ws 2, 3 or 4 (patch_ar_4rank.py); which of them has a kernel is r4d.select's answer\n"
    "            if \"tp\" in getattr(self, \"unique_name\", \"\") and getattr(self, \"world_size\", 1) in (2, 3, 4):\n"
)


def main():
    apply(AR, ALLOC_ANCHOR, ALLOC_NEW, "patch_ar_4rank.py): ws=2 keeps its single region",
          "all-reduce 4-rank: peer list / scratch regions")
    apply(AR, CAP_ANCHOR, CAP_NEW, "patch_ar_4rank.py: decode-size messages only at ws=4",
          "all-reduce 4-rank: ws=4 message cap")
    apply(AR, LAUNCH_ANCHOR, LAUNCH_NEW, "patch_ar_4rank.py: (my scratch, 3 peer scratches",
          "all-reduce 4-rank: launch arms")
    apply(AR, QUANT_ANCHOR, QUANT_NEW, "patch_ar_4rank.py: no compressed 3-/4-rank kernel",
          "all-reduce 4-rank: quant path ws gate")
    apply(AR, _SHADOW_ANCHOR, _SHADOW_NEW, "RADIANCE_AR4_SHADOW", "all-reduce 4-rank: shadow flag")
    apply(AR, SHADOW_ANCHOR, SHADOW_NEW, "patch_ar_4rank.py diagnostic (RADIANCE_AR4_SHADOW=1",
          "all-reduce 4-rank: shadow check method")
    apply(AR, NB_ANCHOR, NB_NEW, "RADIANCE_AR4_NB (default 8)", "all-reduce 4-rank: fixed block count")
    apply(AR, GATE_ANCHOR, GATE_NEW, "ws 2, 3 or 4 (patch_ar_4rank.py)",
          "all-reduce 4-rank: install gate")


if __name__ == "__main__":
    main()
