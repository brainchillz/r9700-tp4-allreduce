#!/bin/bash
# boot only (no bench): tp4-boot.sh LABEL  (env passes through: RADIANCE_USE_R4D_AR, VLLM_TORCH_PROFILER_DIR, RADIANCE_STEP_TRACE)
LABEL=${1:-tp4}
docker rm -f vllmtp4 >/dev/null 2>&1
nohup ~/bench/tp4-ar4-launch.sh > ~/bench/$LABEL-launch.log 2>&1 &
for i in $(seq 1 120); do sleep 10; curl -sf localhost:8090/health >/dev/null 2>&1 && { echo "healthy after $((i*10)) s"; exit 0; }; done
echo "not healthy"; exit 1
