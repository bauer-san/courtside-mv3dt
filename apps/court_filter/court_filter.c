/* Courtside: per-camera court filter between the primary detector and the tracker.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Config file ($COURT_FILTER_CONFIG), whitespace separated:
 *   xmin xmax ymin ymax frame_height        court rectangle incl. margin (world metres), streammux height
 *   <source_id> p00 p01 ... p23             one line per source: 3x4 world->pixel matrix, row-major
 * Pixel coordinates are those of the streammux output (rect_params); the matrices must be valid at
 * that resolution (1920x1080 for the court cameras).
 */
#include <math.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "court_filter.h"

#define CF_MAX_SOURCES 64

typedef struct
{
  gboolean valid;
  double hinv[9];   /* inverse of P columns 0,1,3 (floor homography) */
  double p2[4];     /* third row of P, for the in-front-of-camera test */
} CourtCam;

static CourtCam cams[CF_MAX_SOURCES];
static double xmin, xmax, ymin, ymax, frame_h;
static guint64 n_seen, n_removed;

static gboolean
invert3 (const double m[9], double out[9])
{
  double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
      + m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (fabs (det) < 1e-12)
    return FALSE;
  out[0] = (m[4] * m[8] - m[5] * m[7]) / det;
  out[1] = (m[2] * m[7] - m[1] * m[8]) / det;
  out[2] = (m[1] * m[5] - m[2] * m[4]) / det;
  out[3] = (m[5] * m[6] - m[3] * m[8]) / det;
  out[4] = (m[0] * m[8] - m[2] * m[6]) / det;
  out[5] = (m[2] * m[3] - m[0] * m[5]) / det;
  out[6] = (m[3] * m[7] - m[4] * m[6]) / det;
  out[7] = (m[1] * m[6] - m[0] * m[7]) / det;
  out[8] = (m[0] * m[4] - m[1] * m[3]) / det;
  return TRUE;
}

static gboolean
load_config (const char *path)
{
  FILE *f = fopen (path, "r");
  int n = 0;
  if (!f) {
    g_printerr ("court_filter: cannot open %s\n", path);
    return FALSE;
  }
  if (fscanf (f, "%lf %lf %lf %lf %lf", &xmin, &xmax, &ymin, &ymax, &frame_h) != 5) {
    fclose (f);
    return FALSE;
  }
  int id;
  while (fscanf (f, "%d", &id) == 1) {
    double p[12];
    for (int i = 0; i < 12; i++)
      if (fscanf (f, "%lf", &p[i]) != 1) {
        fclose (f);
        return FALSE;
      }
    if (id < 0 || id >= CF_MAX_SOURCES)
      continue;
    double h[9] = { p[0], p[1], p[3], p[4], p[5], p[7], p[8], p[9], p[11] };
    if (!invert3 (h, cams[id].hinv))
      continue;
    cams[id].p2[0] = p[8];
    cams[id].p2[1] = p[9];
    cams[id].p2[2] = p[10];
    cams[id].p2[3] = p[11];
    cams[id].valid = TRUE;
    n++;
  }
  fclose (f);
  g_print ("court_filter: %d cameras, court x[%.2f, %.2f] y[%.2f, %.2f]\n", n, xmin, xmax, ymin, ymax);
  return n > 0;
}

static gboolean
on_court (const CourtCam * c, double u, double v)
{
  const double *h = c->hinv;
  double w = h[6] * u + h[7] * v + h[8];
  if (fabs (w) < 1e-12)
    return FALSE;
  double x = (h[0] * u + h[1] * v + h[2]) / w;
  double y = (h[3] * u + h[4] * v + h[5]) / w;
  double depth = c->p2[0] * x + c->p2[1] * y + c->p2[3];
  return depth > 0 && x >= xmin && x <= xmax && y >= ymin && y <= ymax;
}

static GstPadProbeReturn
court_filter_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta (GST_PAD_PROBE_INFO_BUFFER (info));
  if (!batch)
    return GST_PAD_PROBE_OK;
  for (NvDsMetaList * lf = batch->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *frame = (NvDsFrameMeta *) lf->data;
    if (frame->source_id >= CF_MAX_SOURCES || !cams[frame->source_id].valid)
      continue;
    NvDsMetaList *lo = frame->obj_meta_list;
    while (lo) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) lo->data;
      lo = lo->next;
      n_seen++;
      double u = obj->rect_params.left + obj->rect_params.width / 2.0;
      double v = obj->rect_params.top + obj->rect_params.height;
      if (v >= frame_h - 5)
        continue;               /* feet below the image edge: keep */
      if (!on_court (&cams[frame->source_id], u, v)) {
        nvds_remove_obj_meta_from_frame (frame, obj);
        n_removed++;
      }
    }
    if (frame->source_id == 0 && frame->frame_num % 900 == 0 && frame->frame_num > 0)
      g_print ("court_filter: frame %u, removed %" G_GUINT64_FORMAT " of %" G_GUINT64_FORMAT
          " detections (%.1f%%)\n", frame->frame_num, n_removed, n_seen,
          n_seen ? 100.0 * n_removed / n_seen : 0.0);
  }
  return GST_PAD_PROBE_OK;
}

gboolean
court_filter_attach (GstPad * pad)
{
  const char *path = g_getenv ("COURT_FILTER_CONFIG");
  if (!path || !*path || !pad)
    return FALSE;
  if (!load_config (path))
    return FALSE;
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, court_filter_probe, NULL, NULL);
  return TRUE;
}
