#!/usr/bin/env bash
# usage: ar-bench.sh GPUS(comma list)   e.g. ar-bench.sh 2,3   or  ar-bench.sh 0,1,2,3
# env: R4D=<libr4d dir> (default rx6), AR_QUANT=0|1, PATCH=1 applies the radiance AR patch chain incl. patch_ar_4rank.py
set -e
GPUS=$1; shift
N=$(( $(tr -cd , <<<"$GPUS" | wc -c) + 1 ))
HIP_IDX=$(seq -s, 0 $((N-1)))
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
R4D=${R4D:-$HOME/.cache/radiance-libr4d/b9e42ab-rx6}
PRE="cp /r4d/r4d.so /opt/vllm/lib/python3.12/site-packages/r4d.so"
if [ "${PATCH:-0}" = 1 ]; then
  PRE="$PRE && cd /patches && python3 patch_ar_maxbytes.py && python3 patch_ar_geometry.py && python3 patch_ar_4rank.py && cd /"
fi
docker rm -f arbench >/dev/null 2>&1 || true
AR_ENV=(); while IFS= read -r kv; do AR_ENV+=(-e "$kv"); done < <(env | grep -E "^(AR_|RADIANCE_AR4_)")
exec docker run --rm --name arbench --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri \
  -e ROCR_VISIBLE_DEVICES="$GPUS" -e HIP_VISIBLE_DEVICES="$HIP_IDX" \
  -e NCCL_PROTO=Simple -e RADIANCE_USE_R4D=1 -e RADIANCE_USE_R4D_AR=1 \
  -e RADIANCE_USE_R4D_AR_QUANT="${AR_QUANT:-0}" -e RADIANCE_AR_MAX_KB=86016 \
  -e RADIANCE_AR4_MAX_KB="${AR4_MAX_KB:-65536}" \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=${PORT:-29511} "${AR_ENV[@]}" "$@" \
  -v "$R4D":/r4d:z -v "$HOME/bench":/bench:z -v "$HOME/vllm-mxfp4":/patches:z \
  --entrypoint bash "$IMAGE" -lc "$PRE && torchrun --nproc_per_node=$N --master_port=${PORT:-29511} /bench/ar_bench.py"
