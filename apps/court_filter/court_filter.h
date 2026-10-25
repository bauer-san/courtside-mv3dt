/* Gym MV3DT: per-camera court filter between the primary detector and the tracker.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef COURT_FILTER_H
#define COURT_FILTER_H

#include <gst/gst.h>

/* Reads $COURT_FILTER_CONFIG (if set) and adds a buffer probe on `pad` (primary GIE src) that removes
 * detections whose bbox bottom-centre, back-projected onto the floor (z = 0) with that source's 3x4
 * world->pixel matrix, falls outside the configured court rectangle. Returns TRUE if the filter is active. */
gboolean court_filter_attach (GstPad * pad);

#endif
