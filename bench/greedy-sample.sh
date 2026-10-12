#!/bin/bash
# greedy-sample.sh LABEL: 3 deterministic prompts, thinking off, temp 0, 300 tokens -> ~/bench/greedy-LABEL.txt
LABEL=$1; OUT=~/bench/greedy-$LABEL.txt; : > $OUT
for p in "Explain how a PCIe switch routes a peer-to-peer write between two GPUs, in detail." "Write a Python function that merges k sorted lists with a heap, then explain its complexity." "Summarize the causes of the 1929 stock market crash in a structured way."; do
  curl -s localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":300,\"temperature\":0,\"seed\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"choices\"][0][\"message\"][\"content\"]);print(\"<<usage\",d[\"usage\"][\"completion_tokens\"],\">>\")" >> $OUT
  echo "=====" >> $OUT
done
wc -w $OUT
