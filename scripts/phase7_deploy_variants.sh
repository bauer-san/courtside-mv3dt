#!/usr/bin/env bash
# Deployment variants on the RTX 3060 Ti, all without display (no OSD, no annotated video):
#   nodisp_r1           FP16, detector every frame
#   int1_r1..r3         FP16, detector every 2nd frame (MV3DT earlyTerminationAge 3: a tentative track may miss frames)
#   int2_r1..r3         FP16, detector every 3rd frame (earlyTerminationAge 4)
#   int8_r1..r3         INT8 detector (own calibration, tools/int8_calibrate.py), every frame
#   int8int1_r1..r3     INT8, every 2nd frame (earlyTerminationAge 3)
#   int8t10_r1..r3      INT8, every frame, detector threshold 0.1 instead of nvinfer's default 0.2
#   js_r1..r3           as int8t10, with JERSEY_SCALING=1 (bilinear jersey crops; now also the configs' default)
# Each run is followed by the jersey-read conversion and appearance extraction the evaluation needs.
# DATASET (default gym_sync, the time-aligned clips) and PREFIX (default s_) set the dataset and the run directories
# (experiments/<PREFIX><run>).
#   ./scripts/phase7_deploy_variants.sh            # all, in this order
#   ./scripts/phase7_deploy_variants.sh int1_r2    # selected runs
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
DATASET="${DATASET:-gym_sync}"
PREFIX="${PREFIX-s_}"
CAL=external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/PeopleNet2.6.3/resnet34_peoplenet_int8_trt10.16.cache

declare -A ENV=(
  [nodisp_r1]="" [int1_r1]="PGIE_INTERVAL=1" [int1_r2]="PGIE_INTERVAL=1" [int1_r3]="PGIE_INTERVAL=1"
  [int2_r1]="PGIE_INTERVAL=2" [int2_r2]="PGIE_INTERVAL=2" [int2_r3]="PGIE_INTERVAL=2"
  [int8_r1]="PGIE_PRECISION=int8" [int8_r2]="PGIE_PRECISION=int8" [int8_r3]="PGIE_PRECISION=int8"
  [int8int1_r1]="PGIE_PRECISION=int8 PGIE_INTERVAL=1" [int8int1_r2]="PGIE_PRECISION=int8 PGIE_INTERVAL=1"
  [int8int1_r3]="PGIE_PRECISION=int8 PGIE_INTERVAL=1"
  [int8t10_r1]="PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1" [int8t10_r2]="PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1"
  [int8t10_r3]="PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1"
  [js_r1]="PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1 JERSEY_SCALING=1"
  [js_r2]="PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1 JERSEY_SCALING=1"
  [js_r3]="PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1 JERSEY_SCALING=1"
)
declare -A EXTRA=(
  [int1_r1]="TargetManagement.earlyTerminationAge=3" [int1_r2]="TargetManagement.earlyTerminationAge=3"
  [int1_r3]="TargetManagement.earlyTerminationAge=3"
  [int2_r1]="TargetManagement.earlyTerminationAge=4" [int2_r2]="TargetManagement.earlyTerminationAge=4"
  [int2_r3]="TargetManagement.earlyTerminationAge=4"
  [int8int1_r1]="TargetManagement.earlyTerminationAge=3" [int8int1_r2]="TargetManagement.earlyTerminationAge=3"
  [int8int1_r3]="TargetManagement.earlyTerminationAge=3"
)
ORDER=(nodisp_r1 int8_r1 int8_r2 int8_r3 int8t10_r1 int8t10_r2 int8t10_r3)   # interval and js runs only on request
RUNS=("${@:-${ORDER[@]}}")

for v in "${RUNS[@]}"; do
  [[ -v ENV[$v] ]] || { echo "unknown run $v"; exit 1; }
  if [[ "${ENV[$v]}" == *int8* && ! -f "$CAL" ]]; then echo "skip $v: no INT8 calibration cache ($CAL)"; continue; fi
  r="$PREFIX$v"
  echo "=== $r: ${ENV[$v]} ${EXTRA[$v]:-} on $DATASET ($(date +%H:%M))"
  env DETECTOR_MODEL=PeopleNet2.6.3 COURT_FILTER=1 JERSEY=1 NEIGHBORS=all DISPLAY_OUT=0 ${ENV[$v]} \
    ./scripts/run_mv3dt.sh "$DATASET" "$r" ${EXTRA[$v]:-} > "experiments/$r.log" 2>&1 || { echo "run $r failed"; continue; }
  "$PY" -m tools.jersey from-ds "$r" > /dev/null
  cp experiments/full_pn263_cf_jersey/{team_model.npz,teams.yml} "experiments/$r/"
  "$PY" -m tools.appearance extract "$r" > /dev/null
  sed -n 2,3p "experiments/$r/run.txt"
done
echo "done ($(date +%H:%M))"
