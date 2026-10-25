#!/usr/bin/env bash
# Phase 6 option A: tune the single-view part of the MV3DT tracker.
#   ./scripts/phase6_tracker_sweep.sh <dataset> <run-prefix> [step ...]
# All steps: PeopleNet 2.6.3 + court filter. Steps without "_mv" switch the multi-view associator off so the
# runs isolate single-camera fragmentation (evaluate: pc_life_med_s, pc_ids_per_min).
# Each step adds a section group from NVIDIA's general-purpose config_tracker_NvDCF_accuracy.yml
# (/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/ in the container) on top of the MV3DT
# template + override_tracker_4cam.yml:
#   base            template as is (MV off)
#   da              + DataAssociator (looser matching; tentativeDetectorConfidence 0.70 -> 0.10)
#   da_tm           + TargetManagement (minTrackerConfidence 0.70 -> 0.15, probationAge 2, minIouDiff4NewTarget)
#   da_tm_vt        + VisualTracker (NvDCF appearance model; the 4cam override disables it)
#   da_tm_vt_reid   + ReID re-association (TAO ReIdentificationNet ONNX)
#   <step>_mv       same with the multi-view associator on (neighbours: $NEIGHBORS, default all)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:?dataset}"; PREFIX="${2:?run prefix}"; shift 2
STEPS=("${@:-base da da_tm da_tm_vt da_tm_vt_reid}")
read -r -a STEPS <<<"${STEPS[*]}"

MV_OFF=(MultiViewAssociator.multiViewAssociatorType=0)
DA=(DataAssociator.dataAssociatorType=0 DataAssociator.associationMatcherType=1
    DataAssociator.minMatchingScore4Overall=0.0222 DataAssociator.minMatchingScore4SizeSimilarity=0.3552
    DataAssociator.minMatchingScore4Iou=0.0548 DataAssociator.minMatchingScore4VisualSimilarity=0.5043
    DataAssociator.matchingScoreWeight4VisualSimilarity=0.3951 DataAssociator.matchingScoreWeight4SizeSimilarity=0.6003
    DataAssociator.matchingScoreWeight4Iou=0.4033 DataAssociator.tentativeDetectorConfidence=0.1024
    DataAssociator.minMatchingScore4TentativeIou=0.2852)
TM=(TargetManagement.minTrackerConfidence=0.1513 TargetManagement.probationAge=2
    TargetManagement.minIouDiff4NewTarget=0.3686)
VT=(VisualTracker.visualTrackerType=2 VisualTracker.useColorNames=1 VisualTracker.useHog=1
    VisualTracker.featureImgSizeLevel=3 VisualTracker.featureFocusOffsetFactor_y=-0.1054
    VisualTracker.filterLr=0.0767 VisualTracker.filterChannelWeightsLr=0.0339 VisualTracker.gaussianSigma=0.5687)
REID=(TrajectoryManagement.reidExtractionInterval=8
      TrajectoryManagement.minMatchingScore4ReidSimilarity=0.0771
      TrajectoryManagement.matchingScoreWeight4ReidSimilarity=0.3848
      ReID.reidType=2 ReID.batchSize=100 ReID.workspaceSize=1000 ReID.reidFeatureSize=256 ReID.reidHistorySize=100
      "ReID.inferDims=[3, 256, 128]" ReID.networkMode=1 ReID.inputOrder=0 ReID.colorFormat=0
      "ReID.offsets=[123.6750, 116.2800, 103.5300]" ReID.netScaleFactor=0.01735207 ReID.keepAspc=1
      ReID.useVPICropScaler=1 ReID.addFeatureNormalization=1 ReID.minVisibility4GalleryUpdate=0.6
      ReID.onnxFile=/workspace/models/ReID/resnet50_market1501_aicity156.onnx
      ReID.modelEngineFile=/workspace/models/ReID/resnet50_market1501_aicity156.onnx_b100_gpu0_fp16.engine)

status=0
for step in "${STEPS[@]}"; do
  base="${step%_mv}"
  args=()
  case "$base" in
    base) ;;
    da) args=("${DA[@]}") ;;
    da_tm) args=("${DA[@]}" "${TM[@]}") ;;
    da_tm_vt) args=("${DA[@]}" "${TM[@]}" "${VT[@]}") ;;
    da_tm_vt_reid) args=("${DA[@]}" "${TM[@]}" "${VT[@]}" "${REID[@]}") ;;
    *) echo "unknown step $step"; status=1; continue ;;
  esac
  [[ "$step" == *_mv ]] || args+=("${MV_OFF[@]}")
  run="${PREFIX}_${step}"
  echo "=== $run ($(date +%H:%M))"
  COURT_FILTER=1 DETECTOR_MODEL=PeopleNet2.6.3 "$ROOT/scripts/run_mv3dt.sh" "$DATASET" "$run" "${args[@]}" \
    >"$ROOT/experiments/$run.log" 2>&1 </dev/null
  rc=$?
  echo "=== $run exit=$rc ($(date +%H:%M)) $(grep -E '^last_perf' "$ROOT/experiments/$run/run.txt" 2>/dev/null)"
  [[ $rc -eq 0 ]] || status=1
done
exit $status
