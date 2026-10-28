#!/usr/bin/env bash
# On this PC: pack what a Jetson needs that isn't in git (README, "Jetson AGX Orin"), into one uncompressed tar:
#   the models under NVIDIA's MV3DT app (PeopleNet 2.6.3 ONNX + labels + INT8 calibration cache, BodyPose3DNet,
#   the PP-OCRv4 jersey models; no TensorRT engines, they are rebuilt on the device), the time-aligned clips
#   (datasets/gym_sync/videos), the INT8 calibration frames, and the team model the run scripts copy into each run.
# Unpack in the repository root on the device: tar -xf <bundle>.tar
#   ./scripts/jetson_bundle.sh /path/to/gym-mv3dt-jetson.tar
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUTTAR="${1:?output tar path}"
M=external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models
FILES=(
  "$M/PeopleNet2.6.3/resnet34_peoplenet.onnx" "$M/PeopleNet2.6.3/detector_labels.txt"
  "$M/PeopleNet2.6.3/resnet34_peoplenet_int8_trt10.16.cache"
  "$M/BodyPose3DNet/bodypose3dnet_accuracy.onnx"
  "$M/OCR/ch_PP-OCRv4_det_infer.onnx" "$M/OCR/ch_PP-OCRv4_rec_infer.onnx"
  datasets/gym_sync/videos
  experiments/int8_calib/calib_frames.npy
  experiments/full_pn263_cf_jersey/team_model.npz experiments/full_pn263_cf_jersey/teams.yml
)
for f in "${FILES[@]}"; do [[ -e "$f" ]] || { echo "missing $f"; exit 1; }; done
tar -cf "$OUTTAR" "${FILES[@]}"
echo "wrote $OUTTAR ($(du -h "$OUTTAR" | cut -f1))"
