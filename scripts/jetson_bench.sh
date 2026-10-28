#!/usr/bin/env bash
# On the Jetson AGX Orin (GitHub issues #4 and #6): the default configuration on the time-aligned clips, as on the
# PC (scripts/run_default.sh), in these groups:
#   bias    agx_cal_r1, agx_cal_r2         camera bias with this device's detector build (not scored)
#   gpu     agx_r1..agx_r3                 detector on the GPU: fps per camera, GPU load, accuracy
#   dla     agx_dla0_r1, agx_dla1_r1       detector on DLA core 0 / 1: fps, GPU load, accuracy
#   live    agx_ct_r1                      court tracker inside the pipeline (COURTTRACK=1): its per-frame time,
#                                          worst frames and the team model's re-clustering (issue #6)
# The court tracker inside the pipeline reads the camera bias from datasets/gym/camInfo at run time; the PC's values
# are used unless the device's are written there first (tools/calib_bias.py agx_cal_r1 agx_cal_r2 --write).
# Copy the run folders back to the PC and score them there (README, "Jetson AGX Orin").
#   ./scripts/jetson_bench.sh bias gpu dla live
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
run() {   # run name, extra environment
  local r="$1"; shift
  echo "=== $r $* ($(date +%H:%M))"
  env "$@" ./scripts/run_default.sh "$r" > "experiments/$r.log" 2>&1 || { echo "run $r failed"; return; }
  sed -n 2,3p "experiments/$r/run.txt"
}
SETS=("$@"); [[ $# -gt 0 ]] || SETS=(bias gpu dla live)
for g in "${SETS[@]}"; do
  case "$g" in
    bias) run agx_cal_r1; run agx_cal_r2 ;;
    gpu)  run agx_r1; run agx_r2; run agx_r3 ;;
    dla)  run agx_dla0_r1 PGIE_DLA_CORE=0; run agx_dla1_r1 PGIE_DLA_CORE=1 ;;
    live) run agx_ct_r1 COURTTRACK=1 ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done
echo "done ($(date +%H:%M))"
