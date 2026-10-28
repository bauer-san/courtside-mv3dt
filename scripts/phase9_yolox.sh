#!/usr/bin/env bash
# YOLOX-m in place of PeopleNet (GitHub issue #10): the default configuration (court filter, jersey reading, all
# camera neighbours, no display, time-aligned clips) with DETECTOR_MODEL=YOLOX-m, INT8
# (apps/yolox/config_pgie_yolox.txt: score threshold 0.25, NMS IoU 0.45, top 20 boxes per frame). MV3DT's gate for
# starting new tracks (DataAssociator.tentativeDetectorConfidence, 0.70 for PeopleNet) is 0.45 on YOLOX's lower score
# scale: at 0.70 about half of YOLOX's on-court boxes never started a track (yx_cal_r1, yx_cal_r1b). Each run is
# followed by the
# jersey-read conversion and appearance extraction.
#   ./scripts/phase9_yolox.sh yx_cal_r2 yx_cal_r3      # camera bias (not scored)
#   ./scripts/phase9_yolox.sh yx_r1 yx_r2 yx_r3
#   python -m tools.calib_bias yx_cal_r2 yx_cal_r3 --into yx_r1 yx_r2 yx_r3   # their camera bias, then score
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
for r in "$@"; do
  echo "=== $r ($(date +%H:%M))"
  DETECTOR_MODEL=YOLOX-m COURT_FILTER=1 JERSEY=1 NEIGHBORS=all DISPLAY_OUT=0 \
    ./scripts/run_mv3dt.sh "${DATASET:-gym_sync}" "$r" DataAssociator.tentativeDetectorConfidence=0.45 \
    > "experiments/$r.log" 2>&1 || { echo "run $r failed"; continue; }
  "$PY" -m tools.jersey from-ds "$r" > /dev/null
  cp experiments/full_pn263_cf_jersey/{team_model.npz,teams.yml} "experiments/$r/"
  "$PY" -m tools.appearance extract "$r" > /dev/null
  sed -n 2,3p "experiments/$r/run.txt"
done
echo "done ($(date +%H:%M))"
