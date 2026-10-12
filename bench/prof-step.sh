#!/bin/bash
# prof-step.sh LABEL: warm, then profile one short thinking-off decode on :8090, collect traces into ~/bench/prof-LABEL/
LABEL=${1:-prof}
OUT=~/bench/prof-$LABEL; rm -rf $OUT; mkdir -p $OUT
PROF=~/.radiance-cache-paro-093-tp4/prof; rm -rf $PROF/*
req() { curl -s localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 200 word story about a lighthouse keeper.\"}],\"max_tokens\":$1,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"usage\"])"; }
echo "warm:"; req 60; req 60
curl -s -X POST localhost:8090/start_profile; echo " started profile $(date +%T)"
echo "profiled:"; req 160
curl -s -X POST localhost:8090/stop_profile; echo " stopped $(date +%T)"
for i in $(seq 1 60); do sleep 5; n=$(ls $PROF 2>/dev/null | wc -l); [ "$n" -ge 4 ] && break; done
sleep 10; ls -la $PROF; cp -r $PROF/* $OUT/ 2>/dev/null; echo "traces in $OUT"
grep -E "RADIANCE step trace|step trace\]|exec_model|out_iv" ~/bench/tp4-$LABEL-launch.log 2>/dev/null | tail -8
