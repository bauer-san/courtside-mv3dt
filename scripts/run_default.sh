#!/usr/bin/env bash
# One run of the pipeline's default configuration (reports/phase7_report.md): PeopleNet 2.6.3 in INT8 at detection
# threshold 0.1 (NMS, top 20 boxes per frame), court filter, jersey reading (bilinear crops), all camera neighbours,
# no display, on the time-aligned clips. Followed by the jersey-read conversion and appearance extraction, so the run
# can be scored right away (tools/multi_eval.py).
# The INT8 calibration cache must match the TensorRT version of the device it runs on: on a new device, make one with
# tools/int8_calibrate.py and point PGIE_INT8_CALIB at it (README, "INT8 on a new device").
# DATASET (default gym_sync) and the variables of run_mv3dt.sh (e.g. COURTTRACK=1) can still be set.
#   ./scripts/run_default.sh RUN [Section.key=value ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
[[ $# -ge 1 ]] || { echo "usage: $0 RUN [Section.key=value ...]"; exit 1; }
r="$1"; shift
DETECTOR_MODEL=PeopleNet2.6.3 PGIE_PRECISION=int8 PGIE_THRESHOLD=0.1 COURT_FILTER=1 JERSEY=1 NEIGHBORS=all \
  DISPLAY_OUT=0 ./scripts/run_mv3dt.sh "${DATASET:-gym_sync}" "$r" "$@"
"$PY" -m tools.jersey from-ds "$r" > /dev/null
cp experiments/full_pn263_cf_jersey/{team_model.npz,teams.yml} "experiments/$r/"
"$PY" -m tools.appearance extract "$r" > /dev/null
sed -n 2,3p "experiments/$r/run.txt"
