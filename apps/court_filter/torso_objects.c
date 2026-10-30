/* Gym MV3DT: add a "torso" child object to each large player box, for the jersey-number SGIEs.
 * SPDX-License-Identifier: Apache-2.0
 *
 * nvinfer SGIEs scale the whole object box to the network input; the jersey number then needs a very large
 * text-detector input (~960x384). Running the text detector on the torso only (12-62 % of the box height)
 * reads as many numbers at half the input size. Probe on the tracker src pad (object ids are final): for
 * player boxes from the PGIE (gie id 1) at least $JERSEY_MIN_H px high (default 150), not cut by the image
 * border, from the sources in $JERSEY_SOURCES (default "0,1,2,3": all cameras), add a
 * child object with unique_component_id TORSO_GIE_ID (4), the parent's object_id (so
 * secondary-reinfer-interval works per player) and no OSD border. jersey_log removes these helper objects
 * after the SGIEs so they never reach the OSD or the message broker.
 *
 * $JERSEY_UPRIGHT=1: upright crops (apps/jersey_upright). The torso is cut from the frame rotated about the player
 * box centre so the body axis is vertical: the angle phi is the image direction of the segment from 0.9 to 1.5 m
 * above the player's floor point (MV3DT's world foot location, else the floor point under the box bottom), with
 * the camera matrices of $COURT_FILTER_CONFIG; |phi| < 5 deg counts as upright. The upright person box is the box of
 * a 0.25 m x 1.85 m cylinder there in the rotated view, scaled so its unrotated twin matches the real box; the
 * torso is 12-62 % of it (as in the offline test, report section 4.9). The torso object carries this rotated
 * rectangle (upright_meta.h) and its axis-aligned box. nvdspreprocess elements in front of the two jersey SGIEs
 * (torso_objects_upright) rotate the crops; a probe behind the text detector turns its text boxes (upright-torso
 * pixels) into rotated rectangles in the frame. The SGIEs then take their input from tensor meta and no longer
 * apply secondary-reinfer-interval, so torso objects are only made every ($JERSEY_REINFER + 1)-th frame a player
 * is eligible (default 6: every 7th).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gstnvdsmeta.h"
#include "torso_objects.h"
#include "upright_meta.h"

#define MAX_SRC 64

static gboolean src_on[MAX_SRC];
static gfloat min_h = 150.f, t0 = 0.12f, t1 = 0.62f, frame_w = 1920.f, frame_h = 1080.f;

/* upright crops */
static gboolean upright;
static gint reinfer = 6;
static double cam_p[MAX_SRC][12];
static gboolean cam_ok[MAX_SRC];
static GHashTable *seen;          /* (source, object id) -> frames eligible */
static guint64 n_torso, n_tilted;

static gboolean
load_cameras (const char *path)
{
  FILE *f = path ? fopen (path, "r") : NULL;
  double skip[5];
  int id, n = 0;
  if (!f || fscanf (f, "%lf %lf %lf %lf %lf", &skip[0], &skip[1], &skip[2], &skip[3], &skip[4]) != 5) {
    if (f)
      fclose (f);
    return FALSE;
  }
  while (fscanf (f, "%d", &id) == 1) {
    double p[12];
    for (int i = 0; i < 12; i++)
      if (fscanf (f, "%lf", &p[i]) != 1) {
        fclose (f);
        return n > 0;
      }
    if (id >= 0 && id < MAX_SRC) {
      memcpy (cam_p[id], p, sizeof p);
      cam_ok[id] = TRUE;
      n++;
    }
  }
  fclose (f);
  return n > 0;
}

static gboolean
project (const double *P, double x, double y, double z, double *u, double *v)
{
  double w = P[8] * x + P[9] * y + P[10] * z + P[11];
  if (w <= 1e-9)
    return FALSE;
  *u = (P[0] * x + P[1] * y + P[2] * z + P[3]) / w;
  *v = (P[4] * x + P[5] * y + P[6] * z + P[7]) / w;
  return TRUE;
}

/* floor point (z = 0) under pixel (u, v): solve P[:, [0, 1, 3]] (x, y, 1) ~ (u, v, 1) */
static gboolean
floor_point (const double *P, double u, double v, double *x, double *y)
{
  double a[4] = { P[0] - u * P[8], P[1] - u * P[9], P[4] - v * P[8], P[5] - v * P[9] };
  double b[2] = { u * P[11] - P[3], v * P[11] - P[7] };
  double det = a[0] * a[3] - a[1] * a[2];
  if (fabs (det) < 1e-12)
    return FALSE;
  *x = (b[0] * a[3] - a[1] * b[1]) / det;
  *y = (a[0] * b[1] - a[2] * b[0]) / det;
  return TRUE;
}

/* Rotated torso rectangle of player box p (see the header comment). FALSE: no geometry (axis-aligned crop). */
static gboolean
torso_rect (guint src, NvDsObjectMeta * p, UprightRect * out)
{
  const NvOSD_RectParams *r = &p->rect_params;
  const double *P = cam_p[src];
  double x, y, u0, v0, u1, v1;
  const float *foot = NULL;
  if (!cam_ok[src])
    return FALSE;
  for (NvDsMetaList * lu = p->obj_user_meta_list; lu; lu = lu->next) {
    NvDsUserMeta *um = (NvDsUserMeta *) lu->data;
    if (um->base_meta.meta_type == NVDS_OBJ_WORLD_FOOT_LOCATION)
      foot = (const float *) um->user_meta_data;
  }
  if (foot) {
    x = foot[0];
    y = foot[1];
  } else if (!floor_point (P, r->left + r->width / 2, r->top + r->height, &x, &y))
    return FALSE;
  if (!project (P, x, y, 0.9, &u0, &v0) || !project (P, x, y, 1.5, &u1, &v1))
    return FALSE;
  double phi = atan2 (u1 - u0, -(v1 - v0)) * 180 / M_PI;
  double cx = r->left + r->width / 2, cy = r->top + r->height / 2;
  double w = r->width, h = r->height;
  if (fabs (phi) < 5)
    phi = 0;
  else {
    /* cylinder model, unrotated and rotated boxes */
    double a = phi * M_PI / 180, c = cos (a), s = sin (a);
    double m0[4] = { 1e9, 1e9, -1e9, -1e9 }, m1[4] = { 1e9, 1e9, -1e9, -1e9 };
    for (int k = 0; k < 32; k++) {
      double ang = 2 * M_PI * (k % 16) / 16, u, v;
      if (!project (P, x + 0.25 * cos (ang), y + 0.25 * sin (ang), k < 16 ? 0 : 1.85, &u, &v))
        return FALSE;
      double ru = c * (u - cx) + s * (v - cy), rv = -s * (u - cx) + c * (v - cy);   /* frame -> upright */
      m0[0] = fmin (m0[0], u);
      m0[1] = fmin (m0[1], v);
      m0[2] = fmax (m0[2], u);
      m0[3] = fmax (m0[3], v);
      m1[0] = fmin (m1[0], ru);
      m1[1] = fmin (m1[1], rv);
      m1[2] = fmax (m1[2], ru);
      m1[3] = fmax (m1[3], rv);
    }
    if (m0[2] - m0[0] < 1 || m0[3] - m0[1] < 1)
      return FALSE;
    w = (m1[2] - m1[0]) * r->width / (m0[2] - m0[0]);
    h = (m1[3] - m1[1]) * r->height / (m0[3] - m0[1]);
  }
  /* torso: rows t0..t1 of the upright box centred on the box centre */
  double a = phi * M_PI / 180, dy = (t0 + t1) / 2 * h - h / 2;
  out->phi = (float) phi;
  out->cx = (float) (cx - sin (a) * dy);
  out->cy = (float) (cy + cos (a) * dy);
  out->w = (float) w;
  out->h = (float) ((t1 - t0) * h);
  return TRUE;
}

/* Every (reinfer + 1)-th eligible frame of a player. */
static gboolean
due (guint src, guint64 id, guint frame)
{
  guint64 key = ((guint64) src << 56) ^ id;
  guint64 *v = g_hash_table_lookup (seen, &key);
  if (!v) {
    guint64 *k = g_new (guint64, 1);
    *k = key;
    v = g_new0 (guint64, 2);
    g_hash_table_insert (seen, k, v);
  }
  v[1] = frame;
  return v[0]++ % (reinfer + 1) == 0;
}

static gboolean
stale (gpointer key, gpointer value, gpointer frame)
{
  return ((guint64 *) value)[1] + 300 < GPOINTER_TO_UINT (frame);
}

static GstPadProbeReturn
torso_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta (GST_PAD_PROBE_INFO_BUFFER (info));
  if (!batch)
    return GST_PAD_PROBE_OK;
  for (NvDsMetaList * lf = batch->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *frame = (NvDsFrameMeta *) lf->data;
    if (frame->source_id >= MAX_SRC || !src_on[frame->source_id])
      continue;
    GList *persons = NULL;
    for (NvDsMetaList * lo = frame->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *o = (NvDsObjectMeta *) lo->data;
      NvOSD_RectParams *r = &o->rect_params;
      if (o->unique_component_id == 1 && o->class_id == 0 && r->height >= min_h && r->top > 5 &&
          r->top + r->height < frame_h - 5 && r->left > 2 && r->left + r->width < frame_w - 2)
        persons = g_list_prepend (persons, o);
    }
    if (upright && frame->frame_num % 1000 == 0)
      g_hash_table_foreach_remove (seen, stale, GUINT_TO_POINTER (frame->frame_num));
    for (GList * l = persons; l; l = l->next) {
      NvDsObjectMeta *p = (NvDsObjectMeta *) l->data;
      UprightRect ur;
      if (upright && !due (frame->source_id, p->object_id, frame->frame_num))
        continue;
      NvDsObjectMeta *t = nvds_acquire_obj_meta_from_pool (batch);
      t->unique_component_id = TORSO_GIE_ID;
      t->class_id = 0;
      t->object_id = p->object_id;
      t->confidence = p->confidence;
      t->rect_params.left = p->rect_params.left;
      t->rect_params.width = p->rect_params.width;
      t->rect_params.top = p->rect_params.top + t0 * p->rect_params.height;
      t->rect_params.height = (t1 - t0) * p->rect_params.height;
      memset (t->misc_obj_info, 0, sizeof t->misc_obj_info);
      if (upright && torso_rect (frame->source_id, p, &ur)) {
        float l, tp, rr, b;
        upright_store (t, &ur);
        upright_aabb (&ur, frame_w, frame_h, &l, &tp, &rr, &b);
        t->rect_params.left = l;
        t->rect_params.top = tp;
        t->rect_params.width = rr - l;
        t->rect_params.height = b - tp;
        n_torso++;
        n_tilted += ur.phi != 0;
      }
      t->rect_params.border_width = 0;
      t->text_params.display_text = NULL;
      g_strlcpy (t->obj_label, "torso", MAX_LABEL_SIZE);
      nvds_add_obj_meta_to_frame (frame, t, p);
    }
    g_list_free (persons);
  }
  return GST_PAD_PROBE_OK;
}

gboolean
torso_objects_attach (GstPad * pad)
{
  if (!pad || !g_getenv ("JERSEY_LOG"))
    return FALSE;
  const char *s = g_getenv ("JERSEY_SOURCES");
  gchar **ids = g_strsplit (s ? s : "0,1,2,3", ",", -1);
  for (gchar ** id = ids; *id; id++) {
    int v = atoi (*id);
    if (v >= 0 && v < MAX_SRC)
      src_on[v] = TRUE;
  }
  g_strfreev (ids);
  if (g_getenv ("JERSEY_MIN_H"))
    min_h = atof (g_getenv ("JERSEY_MIN_H"));
  if (g_getenv ("JERSEY_UPRIGHT") && atoi (g_getenv ("JERSEY_UPRIGHT"))) {
    if (!load_cameras (g_getenv ("COURT_FILTER_CONFIG")))
      g_printerr ("torso_objects: JERSEY_UPRIGHT needs the camera matrices of $COURT_FILTER_CONFIG; crops stay "
          "axis-aligned\n");
    upright = TRUE;
    if (g_getenv ("JERSEY_REINFER"))
      reinfer = atoi (g_getenv ("JERSEY_REINFER"));
    seen = g_hash_table_new_full (g_int64_hash, g_int64_equal, g_free, g_free);
  }
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, torso_probe, NULL, NULL);
  g_print ("torso_objects: torso %.0f-%.0f %% of player boxes >= %.0f px%s\n", 100 * t0, 100 * t1, min_h,
      upright ? ", rotated upright, every 7th eligible frame per player" : "");
  return TRUE;
}

/* Behind the text detector: text boxes in upright-torso pixels -> rotated rectangles in the frame. */
static GstPadProbeReturn
textbox_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  gint det_gie = GPOINTER_TO_INT (data);
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta (GST_PAD_PROBE_INFO_BUFFER (info));
  if (!batch)
    return GST_PAD_PROBE_OK;
  for (NvDsMetaList * lf = batch->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *frame = (NvDsFrameMeta *) lf->data;
    for (NvDsMetaList * lo = frame->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *o = (NvDsObjectMeta *) lo->data;
      UprightRect tr, r;
      if (o->unique_component_id != det_gie || !o->parent || !upright_load (o->parent, &tr))
        continue;
      if (upright_load (o, &r))
        continue;   /* already converted */
      NvOSD_RectParams *b = &o->rect_params;
      float l, t, rr, bb;
      r.phi = tr.phi;
      r.w = b->width;
      r.h = b->height;
      upright_to_frame (&tr, b->left + b->width / 2, b->top + b->height / 2, &r.cx, &r.cy);
      upright_store (o, &r);
      upright_aabb (&r, frame_w, frame_h, &l, &t, &rr, &bb);
      b->left = l;
      b->top = t;
      b->width = rr - l;
      b->height = bb - t;
    }
  }
  return GST_PAD_PROBE_OK;
}

/* Insert nvdspreprocess between `sgie` and whatever feeds it (its queue, or the parent SGIE when that has a single
 * child: deepstream-app then links them directly). */
static gboolean
insert_preprocess (GstElement * sgie, const char *config)
{
  GstElement *bin = GST_ELEMENT (gst_element_get_parent (sgie));
  GstPad *sink = gst_element_get_static_pad (sgie, "sink");
  GstPad *up = sink ? gst_pad_get_peer (sink) : NULL;
  gchar *name = g_strdup_printf ("upright_%s", GST_ELEMENT_NAME (sgie));
  GstElement *pp = gst_element_factory_make ("nvdspreprocess", name);
  gboolean ok = FALSE;
  g_free (name);
  if (!bin || !up || !pp) {
    g_printerr ("torso_objects: cannot insert nvdspreprocess in front of %s\n", GST_ELEMENT_NAME (sgie));
    goto done;
  }
  g_object_set (G_OBJECT (pp), "config-file", config, NULL);
  gst_bin_add (GST_BIN (bin), pp);
  gst_pad_unlink (up, sink);
  GstPad *pp_sink = gst_element_get_static_pad (pp, "sink"), *pp_src = gst_element_get_static_pad (pp, "src");
  ok = gst_pad_link (up, pp_sink) == GST_PAD_LINK_OK && gst_pad_link (pp_src, sink) == GST_PAD_LINK_OK;
  gst_object_unref (pp_sink);
  gst_object_unref (pp_src);
  if (!ok) {
    g_printerr ("torso_objects: cannot link nvdspreprocess in front of %s\n", GST_ELEMENT_NAME (sgie));
    goto done;
  }
  g_object_set (G_OBJECT (sgie), "input-tensor-meta", TRUE, NULL);
  g_print ("torso_objects: %s takes upright crops (%s)\n", GST_ELEMENT_NAME (sgie), config);
done:
  if (up)
    gst_object_unref (up);
  if (sink)
    gst_object_unref (sink);
  if (bin)
    gst_object_unref (bin);
  return ok;
}

void
torso_objects_upright (GstElement ** sgies, guint n)
{
  if (!g_getenv ("JERSEY_LOG") || !g_getenv ("JERSEY_UPRIGHT") || !atoi (g_getenv ("JERSEY_UPRIGHT")))
    return;
  gint det_gie = g_getenv ("JERSEY_DET_GIE") ? atoi (g_getenv ("JERSEY_DET_GIE")) : 2;
  gint rec_gie = g_getenv ("JERSEY_REC_GIE") ? atoi (g_getenv ("JERSEY_REC_GIE")) : 3;
  for (guint i = 0; i < n; i++) {
    if (!sgies[i])
      continue;
    guint uid = 0;
    g_object_get (G_OBJECT (sgies[i]), "unique-id", &uid, NULL);
    if ((gint) uid == det_gie && insert_preprocess (sgies[i], "config_preprocess_textdet.txt")) {
      g_object_set (G_OBJECT (sgies[i]), "clip-object-outside-roi", FALSE, NULL);
      GstPad *src = gst_element_get_static_pad (sgies[i], "src");
      gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER, textbox_probe, GINT_TO_POINTER (det_gie), NULL);
      gst_object_unref (src);
    } else if ((gint) uid == rec_gie)
      insert_preprocess (sgies[i], "config_preprocess_textrec.txt");
  }
}

void
torso_objects_report (void)
{
  if (upright)
    g_print ("torso_objects: %" G_GUINT64_FORMAT " upright torso crops, %" G_GUINT64_FORMAT " of them rotated\n",
        n_torso, n_tilted);
}

void
torso_objects_retarget (GstElement ** sgies, guint n)
{
  if (!g_getenv ("JERSEY_LOG"))
    return;
  gint det_gie = g_getenv ("JERSEY_DET_GIE") ? atoi (g_getenv ("JERSEY_DET_GIE")) : 2;
  for (guint i = 0; i < n; i++) {
    if (!sgies[i])
      continue;
    guint uid = 0;
    g_object_get (G_OBJECT (sgies[i]), "unique-id", &uid, NULL);
    if ((gint) uid == det_gie) {
      g_object_set (G_OBJECT (sgies[i]), "infer-on-gie-id", TORSO_GIE_ID, NULL);
      g_print ("torso_objects: secondary GIE %u now infers on torso objects (gie %d)\n", uid, TORSO_GIE_ID);
    }
  }
}
