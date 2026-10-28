#!/usr/bin/env bash
# Run MV3DT headless on a court dataset.
#   ./scripts/run_mv3dt.sh <dataset> <run-name> [Section.key=value ...]
#   e.g. ./scripts/run_mv3dt.sh gym_60s c60_pnt
#        COURT_FILTER=1 ./scripts/run_mv3dt.sh gym_60s c60_rtdetr_cf   # drop off-court detections before the tracker
#        NEIGHBORS=pairs ./scripts/run_mv3dt.sh ...   # camera neighbour topology (tools/pub_sub.py): all|pairs|pairs_centre
#        JERSEY=1 ./scripts/run_mv3dt.sh ...          # jersey-number SGIEs (apps/jersey) + jersey_reads.csv
#        JERSEY_SCALING=1 (with JERSEY=1)              # bilinear crop scaling for the jersey SGIEs (default: nvinfer's)
#        DETECTOR_MODEL=PeopleNet2.6.3 ./scripts/run_mv3dt.sh gym_60s c60_pn263
#        ./scripts/run_mv3dt.sh gym_60s c60_nopose PoseEstimator.poseEstimatorType=0
#
# Configs come from NVIDIA's auto-configurator (their templates + override_tracker_4cam.yml, all 3 other
# cameras as vision neighbours, streammux/tracker at the 1920x1080 video resolution), then patched:
#   tracker: foot location + visibility output, terminated-track dumps (foot world/image position per stream)
#            (maxTrajectoryBufferLength must be > 0 with terminated-track output: -1 crashes with std::bad_alloc)
#   pipeline: MP4 file sink (hardware encoder) instead of the on-screen display
# Extra Section.key=value arguments patch config_tracker.yml (used by the Phase 6 variants).
#
# Outputs in experiments/<run-name>/:
#   config_*.txt/yml, pub_sub_info_config_0.yml, camInfo_used/, outVideos/osd.mp4, bev/*.mp4, track_dump/track_dump_<stream>.txt,
#   kafka.jsonl.gz, console.log, gpu.csv, run.txt
# Per-frame KITTI dumps are disabled: one file per camera per frame (>200k files for 15 min) stalls the
# HDD/NTFS writeback; kafka.jsonl.gz and track_dump/ carry the same data.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
PY="${MV3DT_VENV:-$HOME/.venvs/mv3dt}/bin/python"
APPPY="$APP/mv3dt_venv/bin/python"

DATASET="${1:?dataset name}"; RUN="${2:?run name}"; shift 2
DS="$ROOT/datasets/$DATASET"
OUT="$ROOT/experiments/$RUN"
DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"
case "$DETECTOR_MODEL" in
  RTDETR) DETECTOR_CONFIG=config_pgie_rt_detr.txt ;;
  PeopleNet2.6.3) DETECTOR_CONFIG=config_pgie_peoplenet.txt ;;
  *) DETECTOR_CONFIG=config_pgie.txt ;;
esac
[[ -d "$DS/videos" ]] || { echo "no dataset $DS"; exit 1; }
[[ -e "$OUT" ]] && { echo "$OUT exists; choose a new run name"; exit 1; }
mkdir -p "$OUT"/{outVideos,bev,track_dump}

"$ROOT/scripts/brokers.sh" start

# 1. NVIDIA auto-configurator (must run from the app dir: templates are relative paths).
# It shells out to `python` for the pub/sub generator, so mv3dt_venv must be on PATH (NVIDIA activates it).
( cd "$APP" && PATH="$APP/mv3dt_venv/bin:$PATH" python utils/deepstream_auto_configurator.py \
    --dataset-dir="$DS" --enable-msg-broker --enable-file-output \
    --detector-config="$DETECTOR_CONFIG" --tracker-config=config_tracker.yml \
    --config-overrides=override_tracker_4cam.yml --output-dir="$OUT" ) | tee "$OUT/configurator.log"

[[ -f "$OUT/pub_sub_info_config_0.yml" ]] || { echo "pub_sub_info_config_0.yml not generated; see configurator.log"; exit 1; }
if [[ -n "${NEIGHBORS:-}" && "${NEIGHBORS}" != all ]]; then
  mv "$OUT/pub_sub_info_config_0.yml" "$OUT/pub_sub_info_config_0.generated.yml"
  "$PY" -m tools.pub_sub "$NEIGHBORS" "$OUT/pub_sub_info_config_0.yml"
fi

# 2. Patch generated configs.
"$PY" -m tools.ds_config tracker "$OUT/config_tracker.yml" \
  ObjectModelProjection.outputFootLocation=1 ObjectModelProjection.outputVisibility=1 \
  TargetManagement.outputTerminatedTracks=1 TargetManagement.maxTrajectoryBufferLength=1000 \
  TargetManagement.terminatedTrackFilename=/workspace/experiments/track_dump/track_dump "$@"
"$PY" -m tools.ds_config ini "$OUT/config_deepstream.txt" sink2.output-file=outVideos/osd.mp4
# DISPLAY=0: no on-screen display and no annotated video (embedded deployment); Kafka/BEV output unchanged
if [[ "${DISPLAY_OUT:-1}" == 0 ]]; then
  "$PY" -m tools.ds_config ini "$OUT/config_deepstream.txt" osd.enable=0 sink2.enable=0
fi
# PGIE_INTERVAL=n: the detector runs on every (n+1)-th frame; the tracker fills in the frames in between
if [[ -n "${PGIE_INTERVAL:-}" ]]; then
  "$PY" -m tools.ds_config ini "$OUT/config_deepstream.txt" primary-gie.interval="$PGIE_INTERVAL"
fi
# PGIE_THRESHOLD=x: detector confidence threshold for all classes (nvinfer pre-cluster-threshold; default 0.2)
if [[ -n "${PGIE_THRESHOLD:-}" ]]; then
  "$PY" -m tools.ds_config ini "$OUT/config_pgie.txt" class-attrs-all.pre-cluster-threshold="$PGIE_THRESHOLD"
fi
# PGIE_PRECISION=int8: detector in INT8. Calibration cache made with this container's TensorRT (10.16) by
# tools/int8_calibrate.py; NVIDIA's NGC cache (resnet34_peoplenet_int8.txt, TensorRT 10.3) names the activations
# differently, so most layers fell back to FP16/FP32 (tested 2026-10-24). PGIE_INT8_CALIB overrides the cache.
if [[ "${PGIE_PRECISION:-fp16}" == int8 ]]; then
  [[ "$DETECTOR_MODEL" == PeopleNet2.6.3 ]] || { echo "PGIE_PRECISION=int8 needs DETECTOR_MODEL=PeopleNet2.6.3"; exit 1; }
  CAL="${PGIE_INT8_CALIB:-resnet34_peoplenet_int8_trt10.16.cache}"
  "$PY" -m tools.ds_config ini "$OUT/config_pgie.txt" property.network-mode=1 \
    property.int8-calib-file="/workspace/models/PeopleNet2.6.3/$CAL"
  # the [primary-gie] group names the cached engine and overrides config_pgie.txt: point it at the INT8 engine
  # (nvinfer builds it from the ONNX model + calibration cache on first use under this default name, then reuses
  # it; delete it after changing the calibration cache)
  "$PY" -m tools.ds_config ini "$OUT/config_deepstream.txt" \
    primary-gie.model-engine-file=/workspace/models/PeopleNet2.6.3/resnet34_peoplenet.onnx_b4_gpu0_int8.engine
fi
sed -i '/^kitti-track-output-dir=/d;/^gie-kitti-output-dir=/d' "$OUT/config_deepstream.txt"
cp -rL "$DS/camInfo" "$OUT/camInfo_used"
mux=$(awk '/^\[/{s=($0=="[streammux]")} s&&/^(width|height)=/' "$OUT/config_deepstream.txt" | tr '\n' ' ')
[[ "$mux" == "width=1920 height=1080 " ]] || { echo "streammux is '$mux', not 1920x1080: camInfo matrices would need scaling"; exit 1; }

# Optional court filter (apps/court_filter, built by scripts/build_court_app.sh).
APP_BIN=deepstream-test5-app
ENVS=()
if [[ "${COURT_FILTER:-0}" == 1 ]]; then
  [[ -x "$ROOT/external/bin/deepstream-test5-court-app" ]] || { echo "run scripts/build_court_app.sh first"; exit 1; }
  "$PY" -m tools.court_filter_config "$OUT/camInfo_used" "$OUT/court_filter.txt" --margin "${COURT_MARGIN:-0.75}"
  APP_BIN=/workspace/bin/deepstream-test5-court-app
  ENVS=(-e COURT_FILTER_CONFIG=/workspace/experiments/court_filter.txt)
fi

# Optional jersey-number reading in DeepStream: torso objects (gie 4, apps/court_filter/torso_objects.c) ->
# text-detector SGIE (gie 2) -> digit-recogniser SGIE (gie 3). The text detector is declared on gie 1 so that
# deepstream-app builds it, and re-targeted to the torso objects by the court app.
if [[ "${JERSEY:-0}" == 1 ]]; then
  [[ -f "$ROOT/external/bin/libnvdsinfer_jersey.so" ]] || { echo "run scripts/build_court_app.sh first"; exit 1; }
  cp "$ROOT"/apps/jersey/config_sgie_text{det,rec}.txt "$OUT/"
  if [[ -n "${JERSEY_SCALING:-}" ]]; then   # nvinfer scaling-filter for the torso and text crops (1 = bilinear)
    "$PY" -m tools.ds_config ini "$OUT/config_sgie_textdet.txt" property.scaling-filter="$JERSEY_SCALING"
    "$PY" -m tools.ds_config ini "$OUT/config_sgie_textrec.txt" property.scaling-filter="$JERSEY_SCALING"
  fi
  "$PY" -m tools.ds_config ini "$OUT/config_deepstream.txt" \
    secondary-gie0.enable=1 secondary-gie0.gpu-id=0 secondary-gie0.gie-unique-id=2 secondary-gie0.operate-on-gie-id=1 \
    secondary-gie0.operate-on-class-ids=0 secondary-gie0.batch-size=16 secondary-gie0.config-file=config_sgie_textdet.txt \
    secondary-gie1.enable=1 secondary-gie1.gpu-id=0 secondary-gie1.gie-unique-id=3 secondary-gie1.operate-on-gie-id=2 \
    secondary-gie1.operate-on-class-ids=0 secondary-gie1.batch-size=32 secondary-gie1.config-file=config_sgie_textrec.txt
  APP_BIN=/workspace/bin/deepstream-test5-court-app
  ENVS+=(-e JERSEY_LOG=/workspace/experiments/jersey_reads.csv)
fi

# Optional court tracker inside DeepStream (apps/courttrack): court tracker + online team model on every frame, one
# Kafka message per frame on topic "courttrack" (recorded to courttrack.jsonl.gz), identities and tracker inputs logged
# (courttrack_identities.csv, courttrack_inputs.bin). Needs JERSEY=1: the reads come from jersey_log.
if [[ "${COURTTRACK:-0}" == 1 ]]; then
  [[ "${JERSEY:-0}" == 1 ]] || { echo "COURTTRACK=1 needs JERSEY=1"; exit 1; }
  "$PY" -m tools.courttrack_config "$OUT/camInfo_used" "$OUT/courttrack.txt" || exit 1
  ENVS+=(-e COURTTRACK_CONFIG=/workspace/experiments/courttrack.txt)
fi

# 3. Mounts: resolve symlinks (datasets link into inputs/).
MOUNTS=(-v "$ROOT/external/bin:/workspace/bin:ro" -v "$APP/models:/workspace/models" -v "$OUT:/workspace/experiments" -v "$(readlink -f "$DS/camInfo"):/workspace/inputs/camInfo:ro")
for v in "$DS"/videos/*.mp4; do MOUNTS+=(-v "$(readlink -f "$v"):/workspace/inputs/videos/$(basename "$v"):ro"); done

# 4. Recorders (start before the pipeline; topic retention is 30 s).
"$APPPY" "$ROOT/tools/kafka_record.py" --app "$APP" --out "$OUT/kafka.jsonl.gz" >"$OUT/kafka_record.log" 2>&1 &
REC_PID=$!
CT_PID=
if [[ "${COURTTRACK:-0}" == 1 ]]; then
  "$APPPY" "$ROOT/tools/kafka_record.py" --app "$APP" --topic courttrack --json --out "$OUT/courttrack.jsonl.gz" \
    >"$OUT/courttrack_record.log" 2>&1 &
  CT_PID=$!
fi
( cd "$APP" && "$APPPY" utils/kafka_bev_visualizer.py --offline --from-end \
    --dataset-path="$DS" --msgconv-config="$OUT/config_msgconv.txt" --output-path="$OUT/bev" \
    --average-multi-cam --show-ids --first-message-timeout=900 --idle-timeout=60 --max-timeout=14400 ) >"$OUT/bev.log" 2>&1 &
BEV_PID=$!
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.encoder,utilization.decoder,power.draw \
  --format=csv,nounits -l 1 >"$OUT/gpu.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT
sleep 5

# 5. Pipeline.
start=$(date +%s)
CNAME="mv3dt_${RUN}_$$"
# Guard: the container occasionally stays up after "App run successful"; stop it after a grace period.
( until grep -q "App run successful" "$OUT/console.log" 2>/dev/null; do sleep 5; docker ps -q --filter name="$CNAME" | grep -q . || exit 0; done
  sleep 60; docker ps -q --filter name="$CNAME" | grep -q . && { echo "[guard] container still up 60 s after app exit; stopping"; docker stop -t 10 "$CNAME" >/dev/null; } ) &
GUARD_PID=$!
docker run --rm --name "$CNAME" --net=host --runtime=nvidia "${MOUNTS[@]}" "${ENVS[@]}" -w /workspace/experiments "$IMAGE" \
  "$APP_BIN" -c config_deepstream.txt 2>&1 | tee "$OUT/console.log"
rc=${PIPESTATUS[0]}
grep -q "App run successful" "$OUT/console.log" && rc=0   # stopped by the guard after a clean app exit
kill $GUARD_PID 2>/dev/null
elapsed=$(( $(date +%s) - start ))
kill $SMI_PID 2>/dev/null

if [[ $rc -ne 0 ]]; then kill $REC_PID $BEV_PID $CT_PID 2>/dev/null; fi
echo "waiting for Kafka recorder / BEV renderer to finish..."
wait $REC_PID; wait $BEV_PID; [[ -n "$CT_PID" ]] && wait $CT_PID
{
  echo "dataset=$DATASET detector=$DETECTOR_MODEL court_filter=${COURT_FILTER:-0} neighbors=${NEIGHBORS:-all} jersey=${JERSEY:-0} display=${DISPLAY_OUT:-1} pgie_interval=${PGIE_INTERVAL:-0} pgie_precision=${PGIE_PRECISION:-fp16} pgie_threshold=${PGIE_THRESHOLD:-default} courttrack=${COURTTRACK:-0} extra=$*"
  echo "pipeline_exit=$rc pipeline_elapsed_s=$elapsed"
  grep -E "^\*\*PERF:\s+[0-9]" "$OUT/console.log" | tail -1 | sed 's/^/last_perf=/'
} | tee "$OUT/run.txt"
exit $rc
