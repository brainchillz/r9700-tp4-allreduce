#!/bin/bash
# pair-pm.sh: after prof-slowcheck, disable GPU runtime PM (power/control=on, reversible), then fresh boots + bench for ar4 and rccl.
while pgrep -f prof-slowcheck.sh >/dev/null; do sleep 10; done
LOG=~/bench/pair-pm.log; : > $LOG
echo "### runtime PM off $(date +%T)" | tee -a $LOG
for d in 03:00.0 23:00.0 c3:00.0 f3:00.0; do echo on | sudo tee /sys/bus/pci/devices/0000:$d/power/control >/dev/null; done
sleep 3
for d in 03:00.0 23:00.0 c3:00.0 f3:00.0; do echo "$d control=$(cat /sys/bus/pci/devices/0000:$d/power/control) status=$(cat /sys/bus/pci/devices/0000:$d/power/runtime_status)" | tee -a $LOG; done
r0=$(sudo journalctl -k --no-pager --since "-1min" 2>/dev/null | grep -c "PSP is resuming"); echo "resumes in last minute before: $r0" | tee -a $LOG
one() { local L=$1 AR=$2
  echo "### $L $(date +%T)" | tee -a $LOG
  docker rm -f vllmtp4 >/dev/null 2>&1
  RADIANCE_USE_R4D_AR=$AR RADIANCE_STEP_TRACE=50 ~/bench/tp4-boot.sh $L 2>&1 | tee -a $LOG
  curl -sf localhost:8090/health >/dev/null || { echo "boot failed" | tee -a $LOG; return 1; }
  echo "resumes during boot: $(sudo journalctl -k --no-pager --since "-9min" 2>/dev/null | grep -c "PSP is resuming")" | tee -a $LOG
  cd ~/bench
  URL=http://127.0.0.1:8090 MODEL=Qwen3.8 python3 bench_oai.py "$L" 2 2>&1 | tee -a $LOG
  ~/bench/power-during.sh $L 2>&1 | tee -a $LOG
python3 prefill_conc.py 8090 Qwen3.8 2>&1 | grep -E "conc [148]" | tee -a $LOG
  grep -E "gpu_worker.py:1439\] rank0" ~/bench/$L-launch.log | sed -E "s/.*INFO 10-12 ([0-9:]+) .*rank0 steps=([0-9]+).*gpu_span +([0-9.]+)\/.*toks\/step +([0-9.]+).*/\1 steps=\2 gpu_span=\3 toks=\4/" | head -12 | tr "\n" ";" | tee -a $LOG; echo | tee -a $LOG
}
one ar4-pm 1
one rccl-pm 0
docker rm -f vllmtp4 >/dev/null 2>&1
echo "### done $(date +%T)" | tee -a $LOG
