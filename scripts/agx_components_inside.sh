#!/usr/bin/env bash
# Inside the benchmark container (scripts/agx_components.sh): component timings -> experiments/agx_components_<time>/
#   platform       device model, L4T release, TensorRT version
#   detector       PeopleNet 2.6.3 INT8 (FP16 fallback for layers without INT8), batch 4 at 960 x 544: GPU, DLA 0,
#                  DLA 1, and DLA 0 + DLA 1 at once (trtexec; INT8 scales are placeholders, which doesn't change the
#                  speed: the PC's calibration cache is for TensorRT 10.16, JetPack 6 has an older TensorRT)
#   jersey         the PP-OCRv4 text detector (batch 16, 480 x 384) and recogniser (batch 32, 48 x 64), INT8
#   courttrack     the C++ court tracker + online team model replayed on a live run's recorded inputs (ct_replay, -O2)
#   decode         four 1080p clips through the hardware decoder at once (GStreamer nvv4l2decoder), if available
set -uo pipefail
cd /ws
OUT=experiments/agx_components_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
T=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
M=external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models
PN="$M/PeopleNet2.6.3/resnet34_peoplenet.onnx"
log() { echo "$*" | tee -a "$OUT/summary.txt"; }

log "== platform"
log "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
log "L4T: $(head -1 /etc/nv_tegra_release 2>/dev/null || echo none)"
log "TensorRT: $(dpkg-query -W -f='${Version}' libnvinfer10 2>/dev/null || dpkg-query -W -f='${Version}' libnvinfer8 2>/dev/null)"

trt() {   # name, trtexec arguments
  local name="$1"; shift
  "$T" "$@" --warmUp=1000 --duration=20 > "$OUT/$name.log" 2>&1
  local q c fb
  q=$(grep -oP 'Throughput: \K[\d.]+' "$OUT/$name.log")
  c=$(grep -oP 'GPU Compute Time: .*median = \K[\d.]+' "$OUT/$name.log")
  fb=$(grep -ciE 'running on GPU|fallback to GPU|not supported on DLA' "$OUT/$name.log")
  log "$name: ${q:-failed} batches/s, median ${c:-?} ms per batch; DLA-fallback messages: $fb"
}

log "== detector (PeopleNet 2.6.3, batch 4 x 3 x 544 x 960)"
S="--shapes=input_1:0:4x3x544x960"
trt pn_gpu --onnx="$PN" "$S" --int8 --fp16
trt pn_dla0 --onnx="$PN" "$S" --int8 --fp16 --useDLACore=0 --allowGPUFallback
trt pn_dla1 --onnx="$PN" "$S" --int8 --fp16 --useDLACore=1 --allowGPUFallback
if grep -q Throughput "$OUT/pn_dla0.log" && grep -q Throughput "$OUT/pn_dla1.log"; then
  trt pn_dla0_both --onnx="$PN" "$S" --int8 --fp16 --useDLACore=0 --allowGPUFallback &
  trt pn_dla1_both --onnx="$PN" "$S" --int8 --fp16 --useDLACore=1 --allowGPUFallback
  wait
fi

log "== jersey reader"
trt ocr_det --onnx="$M/OCR/ch_PP-OCRv4_det_infer.onnx" --shapes=x:16x3x480x384 --int8 --fp16
trt ocr_rec --onnx="$M/OCR/ch_PP-OCRv4_rec_infer.onnx" --shapes=x:32x3x48x64 --int8 --fp16

log "== court tracker (replay of experiments/ctO2_r1)"
if g++ -std=c++17 -O2 -ffp-contract=off -o "$OUT/ct_replay" apps/courttrack/ct_replay.cpp apps/courttrack/courttrack.cpp \
     apps/courttrack/online_teams.cpp 2> "$OUT/ct_build.log"; then
  "$OUT/ct_replay" experiments/ctO2_r1/courttrack_inputs.bin "$OUT/ct_out.txt" > "$OUT/ct_replay.log" 2>&1
  tail -2 "$OUT/ct_replay.log" | tee -a "$OUT/summary.txt"
else
  log "ct_replay build failed (see ct_build.log)"
fi

log "== decode (4 x 1080p H.264, datasets/gym_60s, 1,800 frames each)"
if gst-inspect-1.0 nvv4l2decoder > /dev/null 2>&1; then
  P=""
  for c in datasets/gym_60s/videos/cam_0*.mp4; do
    P+=" filesrc location=$c ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink sync=false"
  done
  t0=$(date +%s.%N)
  gst-launch-1.0 -q $P > "$OUT/decode.log" 2>&1 && t1=$(date +%s.%N) && \
    log "decode: $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.0f", 7200 / (b - a) }') frames/s over the four streams" \
    || log "decode failed (see decode.log)"
else
  log "decode: nvv4l2decoder not available in this container (skipped)"
fi
chown -R "${HOST_UID:-0}:${HOST_GID:-0}" "$OUT" 2>/dev/null
log "results in $OUT"
