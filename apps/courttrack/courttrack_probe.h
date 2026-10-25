/* Gym MV3DT: the court tracker inside DeepStream (deepstream-test5-court-app).
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef COURTTRACK_PROBE_H
#define COURTTRACK_PROBE_H

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reads $COURTTRACK_CONFIG (if set; written by tools/courttrack_config.py) and adds a buffer probe on `pad`
 * (behind the secondary GIEs, after jersey_log's probe) that runs the court tracker and the online team model on every
 * frame, sends one Kafka message per frame with every player's identity and court position, and optionally logs the
 * per-detection identities and the tracker's inputs. Returns TRUE if active. */
gboolean courttrack_attach (GstPad * pad);

/* A jersey read of tracker object `object_id` on `source` in `frame` (number "" = text found, no digits). Called by
 * jersey_log for every text box; the most confident read per object and frame is used. */
void courttrack_add_read (guint frame, guint source, guint64 object_id, const char *number, float conf);

/* Flush logs and disconnect from Kafka. */
void courttrack_close (void);

#ifdef __cplusplus
}
#endif

#endif
