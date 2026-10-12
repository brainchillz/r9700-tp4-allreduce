#!/bin/bash
# pair-bench.sh: fresh boot + bench_oai + power sampling + prefill_conc, for ar4 then rccl, no profiler.
LOG=~/bench/pair-nodyn.log; : > $LOG
one() { local L=$1 AR=$2
  echo "### $L $(date +%T)" | tee -a $LOG
  docker rm -f vllmtp4 >/dev/null 2>&1
  RADIANCE_USE_R4D_AR=$AR RADIANCE_DYNAMIC_DRAFT=0 RADIANCE_STEP_TRACE=50 ~/bench/tp4-boot.sh $L 2>&1 | tee -a $LOG
  curl -sf localhost:8090/health >/dev/null || { echo "boot failed" | tee -a $LOG; return 1; }
  grep -cE "custom all-reduce INSTALLED" ~/bench/tp4-$L-launch.log | sed "s/^/installed on ranks: /" | tee -a $LOG
  sleep 15; echo "idle power: $(rocm-smi --showpower 2>/dev/null | grep -oE "Power \(W\): [0-9.]+" | awk "{print \$3}" | tr "\n" " ")" | tee -a $LOG
  cd ~/bench
  URL=http://127.0.0.1:8090 MODEL=Qwen3.8 python3 bench_oai.py "$L" 2 2>&1 | tee -a $LOG
  ~/bench/power-during.sh $L 2>&1 | tee -a $LOG
  python3 prefill_conc.py 8090 Qwen3.8 2>&1 | grep -E "pp[0-9]+:|conc [148]" | tee -a $LOG
  grep -E "gpu_worker.py:1439\] rank0" ~/bench/tp4-$L-launch.log | tail -1 | sed -E "s/.*rank0/rank0/" | cut -c1-300 | tee -a $LOG
  grep -iE "SpecDecoding metrics" ~/bench/tp4-$L-launch.log | tail -1 | grep -oE "Mean acceptance length: [0-9.]+" | tee -a $LOG
}
one ar4-nodyn 1
one rccl-nodyn 0
docker rm -f vllmtp4 >/dev/null 2>&1
echo "### done $(date +%T)" | tee -a $LOG
