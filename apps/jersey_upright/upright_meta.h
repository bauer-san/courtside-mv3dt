/* Gym MV3DT: upright jersey crops, geometry handed from torso_objects.c to the preprocess library.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A torso object (and later each text box found on it) carries a rotated rectangle in frame pixels: centre
 * (cx, cy), size (w, h) measured along the upright axes, and the angle phi (degrees, + = the body's top leans to
 * the right in the image). The object's rect_params hold the rectangle's axis-aligned bounding box (clamped to the
 * frame); nvdspreprocess copies that box into its scaling pool and the library rotates it upright from there.
 * Stored in NvDsObjectMeta.misc_obj_info (4 x gint64 = 8 floats), marked with UPRIGHT_MAGIC.
 */
#ifndef UPRIGHT_META_H
#define UPRIGHT_META_H

#include <math.h>
#include <string.h>

#include "nvdsmeta.h"

#define UPRIGHT_MAGIC 0x55505247   /* "UPRG" */

typedef struct
{
  float magic, phi, cx, cy, w, h, pad0, pad1;
} UprightRect;

static inline void
upright_store (NvDsObjectMeta * o, const UprightRect * r)
{
  UprightRect v = *r;
  v.magic = (float) UPRIGHT_MAGIC;
  memcpy (o->misc_obj_info, &v, sizeof v);
}

static inline int
upright_load (const NvDsObjectMeta * o, UprightRect * r)
{
  memcpy (r, o->misc_obj_info, sizeof *r);
  return r->magic == (float) UPRIGHT_MAGIC;
}

/* Upright-rectangle coordinates (origin at its top-left corner) -> frame pixels. */
static inline void
upright_to_frame (const UprightRect * r, float xu, float yu, float *x, float *y)
{
  float a = r->phi * (float) M_PI / 180.f, c = cosf (a), s = sinf (a);
  float dx = xu - r->w / 2, dy = yu - r->h / 2;
  *x = r->cx + c * dx - s * dy;
  *y = r->cy + s * dx + c * dy;
}

/* Axis-aligned bounding box of the rotated rectangle, clamped to a W x H frame. */
static inline void
upright_aabb (const UprightRect * r, float W, float H, float *l, float *t, float *rr, float *b)
{
  float xs[4], ys[4];
  upright_to_frame (r, 0, 0, &xs[0], &ys[0]);
  upright_to_frame (r, r->w, 0, &xs[1], &ys[1]);
  upright_to_frame (r, 0, r->h, &xs[2], &ys[2]);
  upright_to_frame (r, r->w, r->h, &xs[3], &ys[3]);
  *l = *t = 1e9f;
  *rr = *b = -1e9f;
  for (int i = 0; i < 4; i++) {
    *l = fminf (*l, xs[i]);
    *rr = fmaxf (*rr, xs[i]);
    *t = fminf (*t, ys[i]);
    *b = fmaxf (*b, ys[i]);
  }
  *l = fmaxf (0.f, floorf (*l));
  *t = fmaxf (0.f, floorf (*t));
  *rr = fminf (W, ceilf (*rr));
  *b = fminf (H, ceilf (*b));
}

#endif
