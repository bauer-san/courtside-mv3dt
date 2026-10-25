#!/usr/bin/env bash
# Phase 6: tracker / detector variants on the court clips.
#   ./scripts/phase6_variants.sh <dataset> <run-prefix> [variant ...]
#   e.g. ./scripts/phase6_variants.sh courtside_60s c60            # all variants
#        ./scripts/phase6_variants.sh courtside_180s c180 vt_reid
#
# Variants (all start from NVIDIA's config_tracker.yml + override_tracker_4cam.yml, PeopleNetTransformer):
#   nopose    bounding-box anchors only (PoseEstimator off)
#   pose      NVIDIA default: BodyPose3DNet every 30 frames (= the Phase 5 baseline)
#   vt        pose + NvDCF visual tracker restored (the 4cam override disables it); template's tuned params
#   vt_reid   vt + ReID re-association (TAO ReIdentificationNet ONNX; params from NVIDIA's
#             deepstream-tracker-3d config_tracker_NvDCF_accuracy_3D.yml)
#   pn263     pose, detector PeopleNet v2.6.3
#   rtdetr    pose, detector RT-DETR
#   pn263_vt_reid, rtdetr_vt_reid   detector variant + vt + ReID
#   rtdetr_assoc[1], pn263_assoc[1] detector variant + stricter cross-camera fusion (see ASSOC below)
#   *_cf                            same, with the court filter before the tracker (COURT_FILTER=1)
#   *_cf_pairs, *_cf_pairsc         court filter + neighbour topology pairs (NL-FL, NR-FR) / pairs + FL-FR
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:?dataset}"; PREFIX="${2:?run prefix}"; shift 2
VARIANTS=("${@:-nopose pose vt vt_reid pn263 rtdetr}")
read -r -a VARIANTS <<<"${VARIANTS[*]}"

VT=(VisualTracker.visualTrackerType=2 VisualTracker.useColorNames=1 VisualTracker.useHog=1
    VisualTracker.featureImgSizeLevel=5 VisualTracker.featureFocusOffsetFactor_y=-0.10525549278780495
    VisualTracker.filterLr=0.025008995723548734 VisualTracker.filterChannelWeightsLr=0.09460260509799209
    VisualTracker.gaussianSigma=0.44967391866177786)
REID=(TrajectoryManagement.reidExtractionInterval=8
      TrajectoryManagement.minMatchingScore4ReidSimilarity=0.0771
      TrajectoryManagement.matchingScoreWeight4ReidSimilarity=0.3848
      ReID.reidType=2 ReID.batchSize=100 ReID.workspaceSize=1000 ReID.reidFeatureSize=256 ReID.reidHistorySize=100
      "ReID.inferDims=[3, 256, 128]" ReID.networkMode=1 ReID.inputOrder=0 ReID.colorFormat=0
      "ReID.offsets=[123.6750, 116.2800, 103.5300]" ReID.netScaleFactor=0.01735207 ReID.keepAspc=1
      ReID.useVPICropScaler=1 ReID.addFeatureNormalization=1 ReID.minVisibility4GalleryUpdate=0.6
      ReID.onnxFile=/workspace/models/ReID/resnet50_market1501_aicity156.onnx
      ReID.modelEngineFile=/workspace/models/ReID/resnet50_market1501_aicity156.onnx_b100_gpu0_fp16.engine)

# Stricter cross-camera fusion. NVIDIA's override_tracker_4cam.yml (tuned on the synthetic warehouse) uses
# maxPeerToPredDistance4Fusion 4.0 m / minPeerTrackletMatchScore 0.3; on a crowded court that fuses
# different players. assoc = the template defaults; assoc1 = tighter still.
ASSOC=(MultiViewAssociator.maxPeerToPredDistance4Fusion=1.35 MultiViewAssociator.minPeerTrackletMatchScore=0.48)
ASSOC1=(MultiViewAssociator.maxPeerToPredDistance4Fusion=1.0 MultiViewAssociator.minPeerTrackletMatchScore=0.48)

status=0
for v in "${VARIANTS[@]}"; do
  run="${PREFIX}_${v}"
  echo "=== $run ($(date +%H:%M))"
  case "$v" in
    nopose)  "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" PoseEstimator.poseEstimatorType=0 ;;
    pose)    "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    vt)      "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${VT[@]}" ;;
    vt_reid) "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${VT[@]}" "${REID[@]}" ;;
    pn263)   DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    rtdetr)  DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    pn263_vt_reid)  DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${VT[@]}" "${REID[@]}" ;;
    rtdetr_vt_reid) DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${VT[@]}" "${REID[@]}" ;;
    rtdetr_assoc)  DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC[@]}" ;;
    rtdetr_assoc1) DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC1[@]}" ;;
    pn263_assoc)   DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC[@]}" ;;
    pn263_assoc1)  DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC1[@]}" ;;
    rtdetr_cf)        COURT_FILTER=1 DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    rtdetr_assoc_cf)  COURT_FILTER=1 DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC[@]}" ;;
    rtdetr_assoc1_cf) COURT_FILTER=1 DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC1[@]}" ;;
    pn263_cf)         COURT_FILTER=1 DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    pn263_assoc_cf)   COURT_FILTER=1 DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC[@]}" ;;
    pn263_assoc1_cf)  COURT_FILTER=1 DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${ASSOC1[@]}" ;;
    rtdetr_cf_pairs)   NEIGHBORS=pairs COURT_FILTER=1 DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    rtdetr_cf_pairsc)  NEIGHBORS=pairs_centre COURT_FILTER=1 DETECTOR_MODEL=RTDETR "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    pn263_cf_pairs)    NEIGHBORS=pairs COURT_FILTER=1 DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    pn263_cf_pairsc)   NEIGHBORS=pairs_centre COURT_FILTER=1 DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" ;;
    *) echo "unknown variant $v"; status=1; continue ;;
  esac >"$ROOT/experiments/$run.log" 2>&1 </dev/null
  rc=$?
  echo "=== $run exit=$rc ($(date +%H:%M)) $(grep -E '^last_perf' "$ROOT/experiments/$run/run.txt" 2>/dev/null)"
  [[ $rc -eq 0 ]] || status=1
done
exit $status
