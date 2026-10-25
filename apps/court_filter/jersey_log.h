/* Gym MV3DT: CSV log of jersey-number reads from the secondary GIEs.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef JERSEY_LOG_H
#define JERSEY_LOG_H

#include <gst/gst.h>

/* If $JERSEY_LOG is set, add a probe on `pad` (secondary GIE bin src) writing one CSV line per text box. */
gboolean jersey_log_attach (GstPad * pad);
void jersey_log_close (void);

#endif
