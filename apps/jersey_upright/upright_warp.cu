/* Gym MV3DT: rotate a crop upright into a network input tensor (one CUDA kernel).
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>

#include "upright_warp.h"

/* One thread per network pixel: network (u, v) -> upright rectangle (xu, yu) = ((u - ox) / s, (v - oy) / s);
 * outside the rectangle: padding (pixel value 0, as nvinfer pads); inside: -> frame (rotation about the
 * rectangle's centre) -> scaling-pool pixel ((x - roi_l) * rx, (y - roi_t) * ry), bilinear, clamped to the pool
 * crop. Output planar BGR, (p - 127.5) / 127.5 (the reader models' normalisation). */
__global__ void
upright_warp_kernel (const unsigned char *pool, int pitch, int pool_w, int pool_h, UprightWarp w, float *out,
    int W, int H)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x, v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= W || v >= H)
    return;
  float px[3] = { 0.f, 0.f, 0.f };
  float xu = (u + 0.5f - w.ox) / w.s - 0.5f, yu = (v + 0.5f - w.oy) / w.s - 0.5f;
  if (xu >= -0.5f && yu >= -0.5f && xu <= w.rw - 0.5f && yu <= w.rh - 0.5f) {
    float dx = xu + 0.5f - w.rw / 2, dy = yu + 0.5f - w.rh / 2;
    float x = w.cx + w.c * dx - w.sn * dy, y = w.cy + w.sn * dx + w.c * dy;
    float qx = (x - w.roi_l) * w.rx - 0.5f, qy = (y - w.roi_t) * w.ry - 0.5f;
    qx = fminf (fmaxf (qx, 0.f), w.crop_w - 1.f);
    qy = fminf (fmaxf (qy, 0.f), w.crop_h - 1.f);
    int x0 = (int) qx, y0 = (int) qy;
    int x1 = min (x0 + 1, (int) w.crop_w - 1), y1 = min (y0 + 1, (int) w.crop_h - 1);
    float fx = qx - x0, fy = qy - y0;
    const unsigned char *r0 = pool + (size_t) y0 * pitch, *r1 = pool + (size_t) y1 * pitch;
    for (int k = 0; k < 3; k++) {
      float a = r0[4 * x0 + k] * (1 - fx) + r0[4 * x1 + k] * fx;
      float b = r1[4 * x0 + k] * (1 - fx) + r1[4 * x1 + k] * fx;
      px[k] = a * (1 - fy) + b * fy;
    }
  }
  /* pool is RGBA; the models take BGR */
  size_t plane = (size_t) W * H, i = (size_t) v * W + u;
  out[i] = (px[2] - 127.5f) / 127.5f;
  out[plane + i] = (px[1] - 127.5f) / 127.5f;
  out[2 * plane + i] = (px[0] - 127.5f) / 127.5f;
}

cudaError_t
upright_warp (const unsigned char *pool, int pitch, int pool_w, int pool_h, const UprightWarp * w, float *out,
    int W, int H, cudaStream_t stream)
{
  dim3 blk (32, 8), grd ((W + 31) / 32, (H + 7) / 8);
  upright_warp_kernel <<< grd, blk, 0, stream >>> (pool, pitch, pool_w, pool_h, *w, out, W, H);
  return cudaGetLastError ();
}
