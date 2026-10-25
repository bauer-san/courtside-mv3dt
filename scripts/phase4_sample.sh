#!/usr/bin/env bash
# Phase 4: run NVIDIA's 4-camera MV3DT sample exactly as documented (scripts/test_4cam_ds.sh),
# logging GPU use (1 Hz) and the console (incl. **PERF FPS lines) to experiments/phase4_sample/.
# Display access: local connections only (xhost +local:), instead of the README's `sudo xhost +`.
# Usage: [DETECTOR_MODEL=PeopleNet2.6.3|RTDETR] ./scripts/phase4_sample.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
OUT="$ROOT/experiments/phase4_sample/${DETECTOR_MODEL:-PeopleNetTransformer}"
mkdir -p "$OUT"

xhost +local: >/dev/null
"$ROOT/scripts/brokers.sh" start

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.encoder,utilization.decoder,power.draw \
  --format=csv,nounits -l 1 >"$OUT/gpu.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

cd "$APP"
start=$(date +%s)
./scripts/test_4cam_ds.sh 2>&1 | tee "$OUT/console.log"
rc=${PIPESTATUS[0]}
echo "elapsed_s=$(( $(date +%s) - start )) exit=$rc" | tee "$OUT/run.txt"
# Keep the generated configs alongside the logs.
cp -r "$APP/experiments/deepstream/4cam/." "$OUT/configs/" 2>/dev/null || true
exit $rc
