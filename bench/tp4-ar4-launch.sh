#!/bin/bash
# TP=4: int5 PARO Qwen3.8-27B on all four R9700s with the 4-rank one-shot all-reduce (libr4d rx6-ar4 + patch_ar_4rank.py)
# and GPU_MAX_HW_QUEUES=1 (keeps the gfx1201 dispatch fast state; ROCm#6347/#6685). Port 8090, container vllmtp4.
sudo systemctl stop vllm-qwen
docker ps --format '{{.Names}} {{.Status}}'
cd ~/vllm-mxfp4
export RUNTIME=docker MODE=prod MODEL_DIR=Qwen3.8-27B-PARO-int5 SPEC=7 PORT=8090 TP=4 GPUS=0,1,2,3 NAME=vllmtp4 \
  CACHE=$HOME/.radiance-cache-paro-093-tp4 R4D_KEY=${R4D_KEY:-b9e42ab-rx6-ar4} \
  CHAT_TEMPLATE=$HOME/vllm-mxfp4/qwen-fixed-v22.3.jinja SERVED_NAMES="Qwen3.8 Qwen3.8-PARO Qwen3.8-27B"
date
exec ./paroquant/run_paroquant_ar4.sh
