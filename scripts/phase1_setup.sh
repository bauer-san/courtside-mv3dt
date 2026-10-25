#!/usr/bin/env bash
# Phase 1: create the venv (on ext4, outside this NTFS folder), populate datasets/courtside/, run tests.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${MV3DT_VENV:-$HOME/.venvs/mv3dt}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -e ".[dev]"

"$VENV/bin/python" -m tools.setup_workspace
"$VENV/bin/python" -m pytest -q
