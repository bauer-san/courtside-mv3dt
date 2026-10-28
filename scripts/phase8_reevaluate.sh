#!/usr/bin/env bash
# Regenerate every court-tracker result the Phase 7 report quotes, with the current code (CPU only; run while no
# DeepStream run is measuring speed). All runs are on the time-aligned clips (datasets/gym_sync), INT8 detector with
# threshold 0.1, no display, except the alignment comparison (UNALIGNED: the original clips; DOUBLED: the offsets
# doubled).
#   JOBS=8 ./scripts/phase8_reevaluate.sh
#   MAIN="..." LIVE="..." UNALIGNED="..." DOUBLED="..." JOBS=8 ./scripts/phase8_reevaluate.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
J="${JOBS:-8}"
read -r -a MAIN <<< "${MAIN:-js_r1 js_r2 js_r3}"
read -r -a LIVE <<< "${LIVE:-ctj_r1 ctj_r2 ctj_r3}"
read -r -a UNALIGNED <<< "${UNALIGNED:-j_int8t10_r1 j_int8t10_r2 j_int8t10_r3}"
read -r -a DOUBLED <<< "${DOUBLED:-j2x_int8t10_r1}"
step() { echo "=== $* ($(date +%H:%M))"; }

step "identity accuracy (section 4.2)"
"$PY" -m tools.multi_eval "${MAIN[@]}" --configs "default (identity output)" --jobs "$J" --md reports/multi_eval_final.md > /dev/null
"$PY" -m tools.loss_breakdown "${MAIN[@]}" --md reports/loss_breakdown_final.md > /dev/null
step "court heat map (section 4.3)"
"$PY" -m tools.court_heatmap "${MAIN[0]}" --out reports/figures/court_heatmap.png > reports/court_heatmap_zones.txt
step "team labels crop by crop (section 4.4)"
"$PY" -m tools.team_label_check "${MAIN[@]}" "${LIVE[@]}" 2>/dev/null > reports/team_label_check.txt
step "court tracker inside DeepStream (section 4.6)"
"$PY" -m tools.courttrack_export live "${LIVE[@]}" 2>/dev/null > reports/courttrack_live.txt
step "non-players in the output (section 4.7)"
"$PY" -m tools.nonplayer_eval "${MAIN[@]}" "${LIVE[@]}" --jobs 6 --md reports/nonplayer_output.md 2>/dev/null > /dev/null
step "cross-camera position disagreement (section 5)"
for r in "${MAIN[@]}"; do for n in game_2to3 game_4to5; do
  "$PY" -m tools.xcam "$r" --name "$n" 2>/dev/null > "reports/xcam_${r}_${n}.txt"
done; done
step "detection thresholds (section 6.1)"
"$PY" -m tools.deploy_eval --jobs "$J" --md reports/deploy_end_to_end.md > /dev/null
step "time alignment (section 2.4)"
"$PY" -m tools.sync_eval "${UNALIGNED[@]}" "${MAIN[@]}" "${DOUBLED[@]}" --md reports/sync_eval.md 2>/dev/null > /dev/null
step "figures"
"$PY" reports/make_figures.py > /dev/null
step "done"
