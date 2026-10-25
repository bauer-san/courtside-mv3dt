#!/usr/bin/env bash
# Regenerate every court-tracker result the Phase 7 report quotes, with the current code (CPU only; run while no
# DeepStream run is measuring speed). Detection-level tables (deploy_variants_detection.md, deploy_int8_by_category.md)
# and the camera-layout study don't depend on the court tracker and are not regenerated.
#   JOBS=8 ./scripts/phase8_reevaluate.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/.venvs/mv3dt/bin/python"
J="${JOBS:-8}"
MOUNT=(mount_r1 mount_r2 mount_r3)
DEFAULT="default (identity output)"
step() { echo "=== $* ($(date +%H:%M))"; }

step "main configurations (section 4.2)"
"$PY" -m tools.multi_eval "${MOUNT[@]}" --jobs "$J" --md reports/multi_eval_final.md > /dev/null
step "loss breakdown (section 4.4)"
"$PY" -m tools.loss_breakdown "${MOUNT[@]}" --md reports/loss_breakdown_final.md > /dev/null
step "court heat map (section 4.5)"
"$PY" -m tools.court_heatmap mount_r1 --out reports/figures/court_heatmap.png > reports/court_heatmap_zones.txt
step "offline inputs and causal replacements (section 4.6)"
"$PY" -m tools.causal_eval "${MOUNT[@]}" --jobs "$J" --md reports/causal_inputs.md > /dev/null
step "deployment variants (section 6.1)"
declare -A VAR=([nodisp]="nodisp_r1" [int8]="int8_r1 int8_r2 int8_r3" [int8t10]="int8t10_r1 int8t10_r2 int8t10_r3"
                [int1]="int1_r1 int1_r2 int1_r3" [int2]="int2_r1 int2_r2 int2_r3"
                [int8int1]="int8int1_r1 int8int1_r2 int8int1_r3")
for v in nodisp int8 int8t10 int1 int2 int8int1; do
  "$PY" -m tools.multi_eval ${VAR[$v]} --configs "$DEFAULT" --jobs "$J" --md "reports/deploy_idf1_$v.md" > /dev/null
done
"$PY" -m tools.deploy_eval --jobs "$J" --md reports/deploy_end_to_end.md > /dev/null
step "cross-camera position disagreement (section 5)"
for r in "${MOUNT[@]}"; do for n in game_2to3 game_4to5; do
  "$PY" -m tools.xcam "$r" --name "$n" 2>/dev/null > "reports/xcam_${r}_${n}.txt"
done; done
step "court tracker inside DeepStream (section 4.8)"
"$PY" -m tools.courttrack_export live ct2_r1 ct2_r2 ct2_r3 2>/dev/null > reports/courttrack_live.txt
step "figures"
"$PY" reports/make_figures.py > /dev/null
step "done"
