/* Gym MV3DT: log jersey-number reads from the secondary GIEs (text detector -> digit recogniser).
 * SPDX-License-Identifier: Apache-2.0
 *
 * $JERSEY_LOG (if set): CSV path. One line per text box found by the text-detector SGIE (gie id
 * $JERSEY_DET_GIE, default 2): frame, source, parent tracker id, parent bbox, text bbox, number, confidence
 * (number empty when the recogniser SGIE (gie id $JERSEY_REC_GIE, default 3) gave no digit read).
 * Text boxes may hang off a torso helper object (torso_objects.c); the player is then its parent.
 * Each read is also handed to the court tracker (courttrack_probe), whose probe follows this one on the same pad.
 * Afterwards the helper objects (text boxes, torsos) are removed so they never reach the OSD or the broker.
 */
#include <stdio.h>
#include <stdlib.h>

#include "gstnvdsmeta.h"
#include "courttrack_probe.h"
#include "jersey_log.h"
#include "torso_objects.h"

static FILE *out;
static gint det_gie = 2, rec_gie = 3;

static GstPadProbeReturn
jersey_log_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta (GST_PAD_PROBE_INFO_BUFFER (info));
  if (!batch)
    return GST_PAD_PROBE_OK;
  for (NvDsMetaList * lf = batch->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *frame = (NvDsFrameMeta *) lf->data;
    for (NvDsMetaList * lo = frame->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) lo->data;
      if (obj->unique_component_id != det_gie)
        continue;
      const gchar *label = "";
      gfloat prob = 0.f;
      for (NvDsMetaList * lc = obj->classifier_meta_list; lc; lc = lc->next) {
        NvDsClassifierMeta *cm = (NvDsClassifierMeta *) lc->data;
        if (cm->unique_component_id != rec_gie || !cm->label_info_list)
          continue;
        NvDsLabelInfo *li = (NvDsLabelInfo *) cm->label_info_list->data;
        label = li->result_label;
        prob = li->result_prob;
      }
      NvDsObjectMeta *par = obj->parent;
      if (par && par->unique_component_id == TORSO_GIE_ID)
        par = par->parent;
      if (par)
        courttrack_add_read (frame->frame_num, frame->source_id, par->object_id, label, prob);
      fprintf (out, "%u,%u,%" G_GUINT64_FORMAT ",%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s,%.3f\n",
          frame->frame_num, frame->source_id, par ? par->object_id : (guint64) - 1,
          par ? par->rect_params.left : -1, par ? par->rect_params.top : -1,
          par ? par->rect_params.width : -1, par ? par->rect_params.height : -1,
          obj->rect_params.left, obj->rect_params.top, obj->rect_params.width, obj->rect_params.height,
          label, prob);
    }
    GList *helpers = NULL;
    for (NvDsMetaList * lo = frame->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *o = (NvDsObjectMeta *) lo->data;
      if (o->unique_component_id == det_gie || o->unique_component_id == TORSO_GIE_ID)
        helpers = g_list_prepend (helpers, o);
    }
    for (GList * l = helpers; l; l = l->next)
      nvds_remove_obj_meta_from_frame (frame, (NvDsObjectMeta *) l->data);
    g_list_free (helpers);
  }
  fflush (out);    /* the container may be stopped rather than exit cleanly */
  return GST_PAD_PROBE_OK;
}

gboolean
jersey_log_attach (GstPad * pad)
{
  const char *path = g_getenv ("JERSEY_LOG");
  if (!path || !*path || !pad)
    return FALSE;
  if (g_getenv ("JERSEY_DET_GIE"))
    det_gie = atoi (g_getenv ("JERSEY_DET_GIE"));
  if (g_getenv ("JERSEY_REC_GIE"))
    rec_gie = atoi (g_getenv ("JERSEY_REC_GIE"));
  out = fopen (path, "w");
  if (!out) {
    g_printerr ("jersey_log: cannot open %s\n", path);
    return FALSE;
  }
  setvbuf (out, NULL, _IOFBF, 1 << 20);
  fprintf (out, "frame,source,track_id,pl,pt,pw,ph,tl,tt,tw,th,number,conf\n");
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, jersey_log_probe, NULL, NULL);
  g_print ("jersey_log: writing %s (text gie %d, digit gie %d)\n", path, det_gie, rec_gie);
  return TRUE;
}

void
jersey_log_close (void)
{
  if (out) {
    fclose (out);
    out = NULL;
  }
}
