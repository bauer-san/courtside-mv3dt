/* Gym MV3DT: torso child objects for the jersey-number SGIEs.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TORSO_OBJECTS_H
#define TORSO_OBJECTS_H

#include <gst/gst.h>

#define TORSO_GIE_ID 4

/* If $JERSEY_LOG is set, add a probe on `pad` (tracker src) creating torso objects (gie id TORSO_GIE_ID). */
gboolean torso_objects_attach (GstPad * pad);

/* deepstream-app only builds secondary GIEs whose operate-on-gie-id chains back to the primary GIE, so the
 * text detector is configured on gie 1 and re-targeted here to the torso objects: every secondary nvinfer
 * element with unique-id $JERSEY_DET_GIE (default 2) gets infer-on-gie-id = TORSO_GIE_ID. */
void torso_objects_retarget (GstElement ** sgies, guint n);

/* $JERSEY_UPRIGHT=1: insert nvdspreprocess (config_preprocess_textdet.txt / _textrec.txt, apps/jersey_upright)
 * in front of each jersey SGIE, switch the SGIEs to tensor-meta input, and convert the text boxes
 * behind the text detector (see torso_objects.c). */
void torso_objects_upright (GstElement ** sgies, guint n);

/* Counts of upright crops (at exit). */
void torso_objects_report (void);

#endif
