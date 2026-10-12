#!/bin/bash
# shadow-run.sh: eager TP=4 boot with the 4-rank one-shot + shadow RCCL check on every call; a few requests; tally.
LOG=~/bench/shadow-run.log; : > $LOG
echo "### shadow $(date +%T)" | tee -a $LOG
RADIANCE_AR4_SHADOW=1 EXTRA_VLLM_ARGS="--enforce-eager" RADIANCE_STEP_TRACE=50 ~/bench/tp4-boot.sh ar4-shadow 2>&1 | tee -a $LOG
curl -sf localhost:8090/health >/dev/null || { echo "boot failed" | tee -a $LOG; tail -20 ~/bench/ar4-shadow-launch.log | tee -a $LOG; exit 1; }
~/bench/greedy-sample.sh ar4-shadow 2>&1 | tee -a $LOG
curl -s localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 123*45? Think briefly.\"}],\"max_tokens\":300,\"temperature\":0}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"choices\"][0][\"message\"][\"content\"][-80:], d[\"usage\"][\"completion_tokens\"])" | tee -a $LOG
echo "== shadow tally" | tee -a $LOG
grep "AR4 SHADOW" ~/bench/ar4-shadow-launch.log | tail -4 | cut -c1-400 | tee -a $LOG
grep -c "AR4 SHADOW" ~/bench/ar4-shadow-launch.log | tee -a $LOG
grep -iE "SpecDecoding metrics" ~/bench/ar4-shadow-launch.log | tail -2 | grep -oE "Mean acceptance length: [0-9.]+|Per-position acceptance rate: [0-9., ]+" | tee -a $LOG
echo "### done $(date +%T)" | tee -a $LOG
