/* Courtside: add a "torso" child object to each large player box, for the jersey-number SGIEs.
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
 */
#include <stdlib.h>
#include <string.h>

#include "gstnvdsmeta.h"
#include "torso_objects.h"

#define MAX_SRC 64

static gboolean src_on[MAX_SRC];
static gfloat min_h = 150.f, t0 = 0.12f, t1 = 0.62f, frame_w = 1920.f, frame_h = 1080.f;

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
    for (GList * l = persons; l; l = l->next) {
      NvDsObjectMeta *p = (NvDsObjectMeta *) l->data;
      NvDsObjectMeta *t = nvds_acquire_obj_meta_from_pool (batch);
      t->unique_component_id = TORSO_GIE_ID;
      t->class_id = 0;
      t->object_id = p->object_id;
      t->confidence = p->confidence;
      t->rect_params.left = p->rect_params.left;
      t->rect_params.width = p->rect_params.width;
      t->rect_params.top = p->rect_params.top + t0 * p->rect_params.height;
      t->rect_params.height = (t1 - t0) * p->rect_params.height;
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
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, torso_probe, NULL, NULL);
  g_print ("torso_objects: torso %.0f-%.0f %% of player boxes >= %.0f px\n", 100 * t0, 100 * t1, min_h);
  return TRUE;
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
