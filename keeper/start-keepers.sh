#!/bin/bash
# start-keepers.sh: hold one idle HIP context on each R9700 (see nexus gpu-context-keeper). Idempotent.
for i in 0 1 2 3; do
  if ! pgrep -f "gpu-context-keeper" >/dev/null 2>&1 || [ "$(pgrep -fc gpu-context-keeper)" -lt 4 ]; then :; fi
done
if [ "$(pgrep -fc "keeper/gpu-context-keeper$")" -ge 4 ]; then echo "keepers already running: $(pgrep -fc "keeper/gpu-context-keeper$")"; exit 0; fi
pkill -f "keeper/gpu-context-keeper$" 2>/dev/null; sleep 1
for i in 0 1 2 3; do ROCR_VISIBLE_DEVICES=$i nohup ~/keeper/gpu-context-keeper > ~/keeper/keeper-$i.log 2>&1 & done
sleep 3; cat ~/keeper/keeper-*.log; echo "keepers: $(pgrep -fc "keeper/gpu-context-keeper$")"
