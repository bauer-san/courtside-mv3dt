/* Gym MV3DT: nvdspreprocess custom library for the jersey reader: crops rotated upright.
 * SPDX-License-Identifier: Apache-2.0
 *
 * nvdspreprocess (object mode) copies each object's rect_params, the axis-aligned box of a rotated rectangle
 * (upright_meta.h), into its scaling pool; CustomTensorPreparation rotates the rectangle upright from there into
 * the network input. [user-configs] mode:
 *   det  text detector on torso objects: letterbox, centred (nvinfer maintain-aspect-ratio + symmetric-padding);
 *        the ROI meta is rewritten so nvinfer places the text boxes in upright-torso pixels (origin at its top-left
 *        corner), which torso_objects.c turns into rotated rectangles in the frame
 *   rec  digit recogniser on text boxes: scaled to fit, top-left aligned, padded right/bottom (symmetric-padding=0)
 * Objects without the geometry are read axis-aligned (phi = 0, the object's box).
 * $UPRIGHT_DUMP=<dir> (check against the offline test): the first 300 rotated text-detector inputs are written there
 * as raw float32 3x480x384, named <source>_<frame>_<object id>.bin.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "nvdspreprocess_interface.h"
#include "upright_meta.h"
#include "upright_warp.h"

struct CustomCtx
{
  bool det;
  int W, H;
  cudaStream_t stream;
  const char *dump;
  int dumped;
};

extern "C" CustomCtx *
initLib (CustomInitParams initparams)
{
  auto *ctx = new CustomCtx;
  ctx->det = initparams.user_configs["mode"] != "rec";
  const auto &shape = initparams.tensor_params.network_input_shape;    // N;C;H;W
  ctx->H = shape[2];
  ctx->W = shape[3];
  cudaStreamCreateWithFlags (&ctx->stream, cudaStreamNonBlocking);
  ctx->dump = ctx->det ? getenv ("UPRIGHT_DUMP") : nullptr;
  ctx->dumped = 0;
  printf ("upright_preprocess: %s, network input %dx%d\n", ctx->det ? "text detector" : "recogniser", ctx->W, ctx->H);
  return ctx;
}

extern "C" void
deInitLib (CustomCtx * ctx)
{
  cudaStreamDestroy (ctx->stream);
  delete ctx;
}

extern "C" NvDsPreProcessStatus
CustomTensorPreparation (CustomCtx * ctx, NvDsPreProcessBatch * batch, NvDsPreProcessCustomBuf * &buf,
    CustomTensorParams & tensorParam, NvDsPreProcessAcquirer * acquirer)
{
  buf = acquirer->acquire ();
  float *out = (float *) buf->memory_ptr;
  const size_t per = (size_t) 3 * ctx->W * ctx->H;
  auto &rois = tensorParam.seq_params.roi_vector;
  for (size_t i = 0; i < batch->units.size (); i++) {
    NvDsRoiMeta & rm = rois[i];
    UprightRect r;
    if (!rm.object_meta || !upright_load (rm.object_meta, &r)) {
      r.phi = 0;
      r.w = rm.roi.width;
      r.h = rm.roi.height;
      r.cx = rm.roi.left + r.w / 2;
      r.cy = rm.roi.top + r.h / 2;
    }
    UprightWarp w;
    w.s = std::fmin (ctx->W / r.w, ctx->H / r.h);
    if (ctx->det) {
      w.ox = (float) ((ctx->W - (int) std::lround (r.w * w.s)) / 2);
      w.oy = (float) ((ctx->H - (int) std::lround (r.h * w.s)) / 2);
    } else {
      w.ox = w.oy = 0;
    }
    w.rw = r.w;
    w.rh = r.h;
    w.cx = r.cx;
    w.cy = r.cy;
    w.c = std::cos (r.phi * (float) M_PI / 180.f);
    w.sn = std::sin (r.phi * (float) M_PI / 180.f);
    w.rx = rm.scale_ratio_x;
    w.ry = rm.scale_ratio_y;
    w.roi_l = rm.roi.left - rm.offset_left / rm.scale_ratio_x;
    w.roi_t = rm.roi.top - rm.offset_top / rm.scale_ratio_y;
    w.crop_w = rm.offset_left + rm.roi.width * rm.scale_ratio_x;
    w.crop_h = rm.offset_top + rm.roi.height * rm.scale_ratio_y;
    if (upright_warp ((const unsigned char *) batch->units[i].converted_frame_ptr, batch->pitch, 0, 0, &w,
            out + i * per, ctx->W, ctx->H, ctx->stream) != cudaSuccess) {
      printf ("upright_preprocess: kernel launch failed\n");
      acquirer->release (buf);
      return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
    }
    if (ctx->det) {    // nvinfer: box = (net - offset) / scale + roi origin -> upright-torso pixels
      rm.roi.left = 0;
      rm.roi.top = 0;
      rm.scale_ratio_x = rm.scale_ratio_y = w.s;
      rm.offset_left = w.ox;
      rm.offset_top = w.oy;
    }
  }
  if (cudaStreamSynchronize (ctx->stream) != cudaSuccess) {
    acquirer->release (buf);
    return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  }
  for (size_t i = 0; ctx->dump && i < batch->units.size () && ctx->dumped < 300; i++) {
    UprightRect r;
    NvDsObjectMeta *o = batch->units[i].roi_meta.object_meta;
    if (!o || !upright_load (o, &r) || r.phi == 0)
      continue;
    std::vector<float> host (per);
    cudaMemcpy (host.data (), out + i * per, per * sizeof (float), cudaMemcpyDeviceToHost);
    char path[512];
    snprintf (path, sizeof path, "%s/%u_%d_%lu.bin", ctx->dump, batch->units[i].frame_meta->source_id,
        batch->units[i].frame_meta->frame_num, (unsigned long) o->object_id);
    FILE *f = fopen (path, "wb");
    if (f) {
      fwrite (host.data (), sizeof (float), per, f);
      fclose (f);
      ctx->dumped++;
    }
  }
  tensorParam.params.network_input_shape[0] = (int) batch->units.size ();
  return NVDSPREPROCESS_SUCCESS;
}
