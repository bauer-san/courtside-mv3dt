/* Courtside: nvinfer custom parsers for jersey-number reading (text detector runs on torso objects) with PaddleOCR PP-OCRv4 (ONNX, Apache-2.0).
 * SPDX-License-Identifier: Apache-2.0
 *
 * NvDsInferParseCustomDBText    secondary detector on person crops: DB probability map [1, H, W] -> text boxes.
 *                               Threshold, 2x2 dilation, 4-connected components, mean-probability score,
 *                               rectangle "unclip"
 *                               (offset = area * ratio / perimeter, as in DB post-processing), optional vertical
 *                               filter (TORSO_TOP..TORSO_BOTTOM of the crop; off: the input is the torso), top
 *                               MAX_BOXES.
 * NvDsInferParseCustomCTCDigits secondary classifier on those boxes: CTC output [T, 6625] (already softmax) ->
 *                               greedy decode; accept 1-2 characters that are all digits; label = the number,
 *                               confidence = mean of the per-character maximum probabilities.
 */
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {
constexpr float BIN_THRESH = 0.3f;
constexpr float BOX_THRESH = 0.5f;     // RapidOCR config: thresh 0.3, box_thresh 0.5, unclip 1.6, dilation
constexpr float UNCLIP_RATIO = 1.6f;
constexpr int MIN_AREA = 10;
constexpr float TORSO_TOP = 0.0f, TORSO_BOTTOM = 1.0f;   // input is already the torso (torso_objects.c)
constexpr int MAX_BOXES = 3;

constexpr int NUM_CLASSES = 6625;   // blank + 6623 characters (ONNX metadata "character") + space
char digit_of(int idx) {
  switch (idx) {                    // indices of the digits in the PP-OCRv4 character table
    case 26: return '0'; case 93: return '1'; case 25: return '2'; case 94: return '3'; case 632: return '4';
    case 631: return '5'; case 933: return '6'; case 29: return '7'; case 27: return '8'; case 1109: return '9';
    default: return 0;
  }
}
}  // namespace

extern "C" bool NvDsInferParseCustomDBText(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                           NvDsInferNetworkInfo const &networkInfo,
                                           NvDsInferParseDetectionParams const &detectionParams,
                                           std::vector<NvDsInferObjectDetectionInfo> &objectList) {
  if (outputLayersInfo.empty()) return false;
  const NvDsInferLayerInfo &layer = outputLayersInfo[0];
  const float *prob = static_cast<const float *>(layer.buffer);
  const int H = networkInfo.height, W = networkInfo.width;
  // binary map dilated with a 2x2 kernel (as PaddleOCR's use_dilation): joins the digits of one number
  std::vector<unsigned char> bin(H * W, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (prob[y * W + x] >= BIN_THRESH) {
        bin[y * W + x] = 1;
        if (x + 1 < W) bin[y * W + x + 1] = 1;
        if (y + 1 < H) bin[(y + 1) * W + x] = 1;
        if (x + 1 < W && y + 1 < H) bin[(y + 1) * W + x + 1] = 1;
      }
  std::vector<int> label(H * W, -1);
  std::vector<int> stack;
  std::vector<NvDsInferObjectDetectionInfo> boxes;
  int ncomp = 0;
  for (int start = 0; start < H * W; ++start) {
    if (!bin[start] || label[start] >= 0) continue;
    int x0 = W, y0 = H, x1 = -1, y1 = -1, area = 0;
    double sum = 0;
    stack.assign(1, start);
    label[start] = ncomp;
    while (!stack.empty()) {
      int p = stack.back();
      stack.pop_back();
      int y = p / W, x = p % W;
      x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
      sum += prob[p];
      ++area;
      const int nb[4] = {p - 1, p + 1, p - W, p + W};
      const bool ok[4] = {x > 0, x < W - 1, y > 0, y < H - 1};
      for (int k = 0; k < 4; ++k)
        if (ok[k] && label[nb[k]] < 0 && bin[nb[k]]) {
          label[nb[k]] = ncomp;
          stack.push_back(nb[k]);
        }
    }
    ++ncomp;
    float score = static_cast<float>(sum / area);
    if (area < MIN_AREA || score < BOX_THRESH) continue;
    float w = x1 - x0 + 1, h = y1 - y0 + 1;
    float off = (w * h) * UNCLIP_RATIO / (2.f * (w + h));
    float l = std::max(0.f, x0 - off), t = std::max(0.f, y0 - off);
    float r = std::min<float>(W, x1 + 1 + off), b = std::min<float>(H, y1 + 1 + off);
    float cy = 0.5f * (t + b);
    if (cy < TORSO_TOP * H || cy > TORSO_BOTTOM * H || (b - t) < 6) continue;
    NvDsInferObjectDetectionInfo o{};
    o.classId = 0;
    o.left = l; o.top = t; o.width = r - l; o.height = b - t;
    o.detectionConfidence = score;
    boxes.push_back(o);
  }
  std::sort(boxes.begin(), boxes.end(), [](const auto &a, const auto &b) {
    return a.detectionConfidence > b.detectionConfidence;
  });
  if (boxes.size() > MAX_BOXES) boxes.resize(MAX_BOXES);
  objectList.insert(objectList.end(), boxes.begin(), boxes.end());
  return true;
}

extern "C" bool NvDsInferParseCustomCTCDigits(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                              NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                              std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {
  if (outputLayersInfo.empty()) return false;
  const NvDsInferLayerInfo &layer = outputLayersInfo[0];
  const NvDsInferDims &d = layer.inferDims;
  if (d.numDims < 2 || d.d[d.numDims - 1] != NUM_CLASSES) return false;
  const int T = d.d[d.numDims - 2];
  const float *out = static_cast<const float *>(layer.buffer);
  std::string text;
  float conf_sum = 0;
  int prev = -1;
  bool valid = true;
  for (int t = 0; t < T; ++t) {
    const float *row = out + static_cast<size_t>(t) * NUM_CLASSES;
    int idx = static_cast<int>(std::max_element(row, row + NUM_CLASSES) - row);
    if (idx != 0 && idx != prev) {
      char c = digit_of(idx);
      if (!c) valid = false;            // a non-digit character: not a jersey number (e.g. "COURTSIDE")
      else { text.push_back(c); conf_sum += row[idx]; }
    }
    prev = idx;
  }
  if (!valid || text.empty() || text.size() > 2) return true;
  float conf = conf_sum / text.size();
  if (conf < classifierThreshold) return true;
  NvDsInferAttribute a{};
  a.attributeIndex = 0;
  a.attributeValue = std::stoi(text);
  a.attributeConfidence = conf;
  a.attributeLabel = strdup(text.c_str());
  attrList.push_back(a);
  attrString = text;
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDBText);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCTCDigits);
