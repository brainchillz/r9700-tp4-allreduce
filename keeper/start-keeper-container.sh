#!/bin/bash
# Hold one idle HIP context on every R9700 from a container (host hipcc is broken; the Radiance image has torch).
# Idempotent. Stop with: docker rm -f gpukeeper
if docker ps --format "{{.Names}}" | grep -qx gpukeeper; then echo "gpukeeper already running"; exit 0; fi
docker rm -f gpukeeper >/dev/null 2>&1
docker run -d --name gpukeeper --restart unless-stopped --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 -e HIP_VISIBLE_DEVICES=0,1,2,3 --entrypoint python3 stilldeadcode/vllm-radiance:0.9.3 -c "
import torch, time
n = torch.cuda.device_count()
bufs = [torch.ones(64, device=f\"cuda:{i}\") for i in range(n)]
for i in range(n): torch.cuda.synchronize(i)
print(f\"gpu-context-keeper: holding idle contexts on {n} devices\", flush=True)
time.sleep(10**9)
" >/dev/null
sleep 8; docker logs gpukeeper 2>&1 | tail -2; rocm-smi --showpids 2>/dev/null | grep -E "^[0-9]+ " | head -6
