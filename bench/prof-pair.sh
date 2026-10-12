#!/bin/bash
# prof-pair.sh: profile one decode on TP=4 with the 4-rank one-shot (ar4) and with RCCL (rccl); summarize.
LOG=~/bench/prof-pair.log; : > $LOG
run_one() {  # run_one LABEL R4D_AR(0|1)
  local L=$1 AR=$2
  echo "### $L $(date +%T)" | tee -a $LOG
  docker rm -f vllmtp4 >/dev/null 2>&1
  RADIANCE_USE_R4D_AR=$AR PROFILE=1 RADIANCE_STEP_TRACE=20 ~/bench/tp4-boot.sh $L 2>&1 | tee -a $LOG
  curl -sf localhost:8090/health >/dev/null || { echo "boot failed" | tee -a $LOG; return 1; }
  ~/bench/prof-step.sh $L 2>&1 | grep -vE "^\s*$" | tee -a $LOG
  ~/bench/greedy-sample.sh $L 2>&1 | tee -a $LOG
  grep -E "gpu_worker.py:1439\] rank0" ~/bench/tp4-$L-launch.log | tail -1 | cut -c1-330 | tee -a $LOG
  grep -iE "SpecDecoding metrics" ~/bench/tp4-$L-launch.log | tail -1 | cut -c1-200 | tee -a $LOG
  for f in ~/bench/prof-$L/*; do echo "--- trace $(basename $f) $(du -h $f | cut -f1)" | tee -a $LOG; done
  f0=$(ls ~/bench/prof-$L/* 2>/dev/null | grep -vE "capture|AsyncLLM|async" | head -1)
  [ -n "$f0" ] && python3 ~/bench/trace_summary.py "$f0" 22 2>&1 | tee -a $LOG
}
run_one ar4-p 1
run_one rccl-p 0
docker rm -f vllmtp4 >/dev/null 2>&1
echo "### diff of greedy outputs (ar4 vs rccl)" | tee -a $LOG
diff <(sed "s/<<usage.*//" ~/bench/greedy-ar4-p.txt) <(sed "s/<<usage.*//" ~/bench/greedy-rccl-p.txt) | head -20 | tee -a $LOG
echo "### done $(date +%T)" | tee -a $LOG
