#!/bin/bash
# power-during.sh LABEL: sample power + sclk on all 4 cards every 0.5 s during a 400-token greedy decode
LABEL=$1; OUT=~/bench/power-$LABEL.txt; : > $OUT
( curl -s localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 600 word essay on the history of lighthouses.\"}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null ) &
sleep 1.5
for i in $(seq 1 8); do rocm-smi --showpower --showclocks 2>/dev/null | grep -E "Package Power|sclk" | sed -E "s/\s+/ /g" >> $OUT; sleep 0.4; done
wait
echo "power W (all samples, all cards):"; grep -oE "Power \(W\): [0-9.]+" $OUT | awk "{print \$3}" | sort -n | uniq -c | sort -k2 -n | tr "\n" ";"; echo
echo "sclk:"; grep -oE "sclk clock level: [^ ]+ \([0-9]+Mhz\)" $OUT | sort | uniq -c | tr "\n" ";"; echo
