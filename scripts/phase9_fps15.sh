#!/usr/bin/env bash
# Cameras at 15 fps (GitHub issue #5): the default configuration on datasets/gym_sync15 (every 2nd frame of the
# time-aligned clips, make_clips --step 2), with the settings that count frames halved so they last the same time:
# MV3DT shadow tracking 162 -> 81 frames, recently-active age 178 -> 89, pose every 30 -> 15 frames (only with POSE=1;
# the pose model is off by default since issue #5); the jersey reader
# re-reads a player every 3rd frame (0.2 s; at 30 fps every 7th, 0.23 s); torso colours from every 2nd frame
# (0.13 s; at 30 fps every 3rd, 0.1 s). probationAge and earlyTerminationAge count detections and stay.
# Each run is then renumbered to 30 fps frames (tools/frame_step.py) for scoring with the usual tools.
#   ./scripts/phase9_fps15.sh [r15_r1 r15_r2 r15_r3]
#   JERSEY_REINFER=6 ./scripts/phase9_fps15.sh r15j7_r1 r15j7_r2 r15j7_r3   # jersey reader every 7th frame (0.47 s)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
RUNS=("$@"); [[ $# -gt 0 ]] || RUNS=(r15_r1 r15_r2 r15_r3)
for r in "${RUNS[@]}"; do
  echo "=== $r ($(date +%H:%M))"
  DETECTOR_MODEL=PeopleNet2.6.3 PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1 COURT_FILTER=1 JERSEY=1 NEIGHBORS=all \
    DISPLAY_OUT=0 JERSEY_REINFER="${JERSEY_REINFER:-2}" ./scripts/run_mv3dt.sh gym_sync15 "$r" \
    TargetManagement.maxShadowTrackingAge=81 MultiViewAssociator.recentlyActiveAge=89 \
    PoseEstimator.poseInferenceInterval=15 > "experiments/$r.log" 2>&1 \
    || { echo "run $r failed"; continue; }
  "$PY" -m tools.jersey from-ds "$r" > /dev/null
  cp experiments/full_pn263_cf_jersey/{team_model.npz,teams.yml} "experiments/$r/"
  "$PY" -m tools.appearance extract "$r" --step 2 > /dev/null
  "$PY" -m tools.frame_step "$r"
  sed -n 2,3p "experiments/$r/run.txt"
done
echo "done ($(date +%H:%M))"
