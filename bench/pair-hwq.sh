#!/bin/bash
# pair-hwq.sh: after pair-keeper, stop the keeper, then test GPU_MAX_HW_QUEUES=1: ar4 twice, rccl once.
while pgrep -f pair-keeper.sh >/dev/null; do sleep 10; done
LOG=~/bench/pair-hwq.log; : > $LOG
docker rm -f gpukeeper >/dev/null 2>&1; echo "### keeper stopped; GPU_MAX_HW_QUEUES=1 $(date +%T)" | tee -a $LOG
one() { local L=$1 AR=$2
  echo "### $L $(date +%T)" | tee -a $LOG
  docker rm -f vllmtp4 >/dev/null 2>&1
  GPU_MAX_HW_QUEUES=1 RADIANCE_USE_R4D_AR=$AR RADIANCE_STEP_TRACE=50 ~/bench/tp4-boot.sh $L 2>&1 | tee -a $LOG
  curl -sf localhost:8090/health >/dev/null || { echo "boot failed" | tee -a $LOG; tail -5 ~/bench/$L-launch.log | tee -a $LOG; return 1; }
  docker exec vllmtp4 bash -c "cat /proc/1/environ | tr \"\\0\" \"\\n\" | grep GPU_MAX_HW_QUEUES" | tee -a $LOG
  cd ~/bench
  URL=http://127.0.0.1:8090 MODEL=Qwen3.8 python3 bench_oai.py "$L" 2 2>&1 | tee -a $LOG
  python3 prefill_conc.py 8090 Qwen3.8 2>&1 | grep -E "conc [148]" | tee -a $LOG
  grep -E "gpu_worker.py:1439\] rank0" ~/bench/$L-launch.log | sed -E "s/.*INFO 10-12 ([0-9:]+) .*rank0 steps=([0-9]+).*gpu_span +([0-9.]+)\/.*toks\/step +([0-9.]+).*/\1 steps=\2 gpu_span=\3 toks=\4/" | head -8 | tr "\n" ";" | tee -a $LOG; echo | tee -a $LOG
}
one ar4-hwq1a 1
one ar4-hwq1b 1
one rccl-hwq1 0
docker rm -f vllmtp4 >/dev/null 2>&1
echo "### done $(date +%T)" | tee -a $LOG
