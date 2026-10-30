/* Gym MV3DT: rotate a crop upright into a network input tensor.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef UPRIGHT_WARP_H
#define UPRIGHT_WARP_H

#include <cuda_runtime.h>

typedef struct
{
  float s, ox, oy;          /* network pixel = upright pixel * s + (ox, oy) */
  float rw, rh;             /* upright rectangle size (frame pixels) */
  float cx, cy, c, sn;      /* its centre in the frame, cos / sin of phi */
  float roi_l, roi_t;       /* frame -> scaling pool: ((x - roi_l) * rx, (y - roi_t) * ry) */
  float rx, ry;
  float crop_w, crop_h;     /* filled part of the pool image */
} UprightWarp;

#ifdef __cplusplus
extern "C"
#endif
cudaError_t upright_warp (const unsigned char *pool, int pitch, int pool_w, int pool_h, const UprightWarp * w,
    float *out, int W, int H, cudaStream_t stream);

#endif
