#!/usr/bin/env bash
# Court tracker inside DeepStream (apps/courttrack), full game, no display: three runs for live accuracy and speed.
# Each run is followed by the jersey-read conversion and appearance extraction, so the Python reference can be run on
# the same DeepStream run (tools/courttrack_export.py live).
# DATASET: default gym_sync (the time-aligned clips).
#   ./scripts/phase8_courttrack_live.sh [ct_r1 ct_r2 ct_r3]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
DATASET="${DATASET:-gym_sync}"
RUNS=("${@:-ct_r1 ct_r2 ct_r3}")
for r in ${RUNS[@]}; do
  echo "=== $r ($(date +%H:%M))"
  env DETECTOR_MODEL=PeopleNet2.6.3 COURT_FILTER=1 JERSEY=1 NEIGHBORS=all DISPLAY_OUT=0 COURTTRACK=1 \
    ./scripts/run_mv3dt.sh "$DATASET" "$r" > "experiments/$r.log" 2>&1 || { echo "run $r failed"; continue; }
  "$PY" -m tools.jersey from-ds "$r" > /dev/null
  cp experiments/full_pn263_cf_jersey/{team_model.npz,teams.yml} "experiments/$r/"
  "$PY" -m tools.appearance extract "$r" > /dev/null
  sed -n 2,3p "experiments/$r/run.txt"
  grep "courttrack:.*stepped" "experiments/$r/console.log"
done
echo "done ($(date +%H:%M))"
