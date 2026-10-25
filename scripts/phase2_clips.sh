#!/usr/bin/env bash
# Phase 2: 60 s and 180 s frame-synced test clips starting at frame 3600, then run tests.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${MV3DT_VENV:-$HOME/.venvs/mv3dt}/bin/python"

"$PY" -m tools.make_clips --start 3600 --seconds 60 --name courtside_60s
"$PY" -m tools.make_clips --start 3600 --seconds 180 --name courtside_180s
"$PY" -m pytest -q
