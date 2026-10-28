/* Gym MV3DT: nvinfer bounding-box parser for YOLOX (Megvii, Apache-2.0; COCO), person class only (GitHub issue #10).
 * SPDX-License-Identifier: Apache-2.0
 *
 * NvDsInferParseCustomYoloX  output "output" [anchors, 85] per image, raw as exported by YOLOX (no decode in the
 *                            graph): per anchor cx, cy (grid offsets), log w, log h, objectness, 80 class scores
 *                            (sigmoids already applied). Anchors are the grid cells of strides 8, 16 and 32 over the
 *                            network input, in that order. Decoded as YOLOX's demo does:
 *                            centre = (offset + cell) * stride, size = exp(log size) * stride.
 *                            score = objectness x "person" (class 0); boxes with score >= the pre-cluster threshold
 *                            go to nvinfer, which scales them back to the frame and clusters them (cluster-mode=2).
 */
#include <cmath>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {
constexpr int NUM_CLASSES = 80;
constexpr int PERSON = 0;
constexpr int STRIDES[] = {8, 16, 32};
}  // namespace

extern "C" bool NvDsInferParseCustomYoloX(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                          NvDsInferNetworkInfo const &networkInfo,
                                          NvDsInferParseDetectionParams const &detectionParams,
                                          std::vector<NvDsInferObjectDetectionInfo> &objectList) {
  const NvDsInferLayerInfo *layer = nullptr;
  for (auto const &l : outputLayersInfo)
    if (l.layerName && std::strcmp(l.layerName, "output") == 0) layer = &l;
  if (!layer || layer->inferDims.numDims < 2 || layer->inferDims.d[layer->inferDims.numDims - 1] != 5 + NUM_CLASSES)
    return false;
  const float *out = static_cast<const float *>(layer->buffer);
  const float thr = detectionParams.perClassPreclusterThreshold.empty()
                        ? 0.25f : detectionParams.perClassPreclusterThreshold[PERSON];
  const int W = static_cast<int>(networkInfo.width), H = static_cast<int>(networkInfo.height);
  int anchor = 0;
  for (int s : STRIDES) {
    const int gw = W / s, gh = H / s;
    for (int gy = 0; gy < gh; ++gy)
      for (int gx = 0; gx < gw; ++gx, ++anchor) {
        const float *a = out + static_cast<size_t>(anchor) * (5 + NUM_CLASSES);
        const float score = a[4] * a[5 + PERSON];
        if (score < thr) continue;
        const float cx = (a[0] + gx) * s, cy = (a[1] + gy) * s;
        const float w = std::exp(a[2]) * s, h = std::exp(a[3]) * s;
        NvDsInferObjectDetectionInfo o{};
        o.classId = PERSON;
        o.left = std::fmax(cx - w / 2, 0.f);
        o.top = std::fmax(cy - h / 2, 0.f);
        o.width = std::fmin(cx + w / 2, static_cast<float>(W)) - o.left;
        o.height = std::fmin(cy + h / 2, static_cast<float>(H)) - o.top;
        o.detectionConfidence = score;
        if (o.width > 0 && o.height > 0) objectList.push_back(o);
      }
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloX);
