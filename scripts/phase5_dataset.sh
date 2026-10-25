#!/usr/bin/env bash
# Phase 5a: complete datasets/courtside (modelInfo, map.png, transforms.yml), render overlays, run tests.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${MV3DT_VENV:-$HOME/.venvs/mv3dt}/bin/python"
"$PY" -m tools.build_dataset
"$PY" -m tools.validate_overlay --frame 3600
"$PY" -m pytest -q
