#!/bin/bash
# clocks-during.sh LABEL: sample sclk/mclk/power/util every 0.5 s while a 400-token greedy decode runs
LABEL=$1; OUT=~/bench/clocks-$LABEL.txt; : > $OUT
( curl -s localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 600 word essay on the history of lighthouses.\"}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null ) &
sleep 1.2
for i in $(seq 1 6); do
  echo "--- t=$i" >> $OUT
  rocm-smi --showclocks --showpower --showuse 2>/dev/null | grep -E "GPU\[[0-3]\].*(sclk|mclk|fclk|Power \(W\)|GPU use)" >> $OUT
  sleep 0.5
done
wait
grep -E "GPU\[0\]" $OUT | sed -E "s/^GPU\[0\]\s*: //" | sort | uniq -c | sort -rn | head -12
