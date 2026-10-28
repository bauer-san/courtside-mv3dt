#!/usr/bin/env bash
# Build deepstream-test5-court-app: NVIDIA's deepstream-test5-app (DeepStream v9.1.0.2 sources) plus
#   apps/court_filter/court_filter.c  per-camera court filter (probe on the primary GIE src, before the tracker)
#   apps/court_filter/torso_objects.c torso child objects for the jersey SGIEs (probe on the tracker src)
#   apps/court_filter/jersey_log.c    CSV log of jersey-number reads (probe on the secondary GIE bin src)
#   apps/courttrack/                  court tracker + online team model, per-frame Kafka messages (probe after
#                                     jersey_log on the same pad; active with $COURTTRACK_CONFIG)
# and the nvinfer parser libraries apps/jersey/libnvdsinfer_jersey.so (text-detector + digit-recogniser SGIEs)
# and apps/yolox/libnvdsinfer_yolox.so (YOLOX person detector, GitHub issue #10).
# Changes to NVIDIA's code: #includes and two probe attachments after create_pipeline() in main.c, one SRCS line
# in the Makefile. Built inside the DeepStream container; binary and library go to external/bin/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="$ROOT/external/DeepStream"
SRC="$DS/src/apps/sample_apps/deepstream-test5"
DST="$DS/src/apps/sample_apps/deepstream-test5-court"
IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"

rm -rf "$DST" && cp -r "$SRC" "$DST"
cp "$ROOT"/apps/court_filter/{court_filter,jersey_log,torso_objects}.{c,h} "$DST/"
cp "$ROOT"/apps/courttrack/{courttrack,online_teams,courttrack_probe}.{cpp,h} "$DST/"
python3 - "$DST" <<'EOF'
import sys
from pathlib import Path
d = Path(sys.argv[1])
main = d / "deepstream_test5_app_main.c"
s = main.read_text()
inc = '#include "deepstream_test5_app.h"\n'
call = """    if (!create_pipeline (appCtx[i], bbox_generated_probe_after_analytics,
            NULL, perf_cb, overlay_graphics)) {
      NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
      return_value = -1;
      goto done;
    }
"""
quit = '  g_print ("Quitting\\n");\n'
assert s.count(inc) == 1 and s.count(call) == 1 and s.count(quit) == 1, "anchors not found: DeepStream source changed?"
s = s.replace(inc, inc + '#include "court_filter.h"  /* gym-mv3dt */\n#include "jersey_log.h"  /* gym-mv3dt */\n#include "torso_objects.h"  /* gym-mv3dt */\n#include "courttrack_probe.h"  /* gym-mv3dt */\n')
s = s.replace(quit, quit + '  courttrack_close ();  /* gym-mv3dt: flush the court tracker */\n')
s = s.replace(call, call + """    { /* gym-mv3dt: court filter before the tracker */
      GstPad *cf_pad = gst_element_get_static_pad (
          appCtx[i]->pipeline.common_elements.primary_gie_bin.bin, "src");
      court_filter_attach (cf_pad);
      if (cf_pad)
        gst_object_unref (cf_pad);
      { /* gym-mv3dt: text detector -> torso objects */
        GstElement *sg[MAX_SECONDARY_GIE_BINS];
        for (guint k = 0; k < MAX_SECONDARY_GIE_BINS; k++)
          sg[k] = appCtx[i]->pipeline.common_elements.secondary_gie_bin.sub_bins[k].secondary_gie;
        torso_objects_retarget (sg, MAX_SECONDARY_GIE_BINS);
      }
      if (appCtx[i]->pipeline.common_elements.tracker_bin.bin) { /* gym-mv3dt: torso objects */
        GstPad *to_pad = gst_element_get_static_pad (
            appCtx[i]->pipeline.common_elements.tracker_bin.bin, "src");
        torso_objects_attach (to_pad);
        if (to_pad)
          gst_object_unref (to_pad);
      }
      if (appCtx[i]->pipeline.common_elements.secondary_gie_bin.bin) { /* gym-mv3dt: jersey reads */
        /* Probe the pad behind the bin's ghost src: deepstream-app's own probe on the ghost pad (message
         * generation) was added earlier and would otherwise see the helper objects before they are removed. */
        GstPad *ghost = gst_element_get_static_pad (
            appCtx[i]->pipeline.common_elements.secondary_gie_bin.bin, "src");
        GstPad *jl_pad = ghost ? gst_ghost_pad_get_target (GST_GHOST_PAD (ghost)) : NULL;
        jersey_log_attach (jl_pad ? jl_pad : ghost);
        courttrack_attach (jl_pad ? jl_pad : ghost);  /* gym-mv3dt: after jersey_log (it hands over the reads) */
        if (jl_pad)
          gst_object_unref (jl_pad);
        if (ghost)
          gst_object_unref (ghost);
      }
    }
""")
main.write_text(s)
mk = d / "Makefile"
m = mk.read_text()
m = m.replace("APP:= deepstream-test5-app", "APP:= deepstream-test5-court-app")
m = m.replace("SRCS:= deepstream_test5_app_main.c deepstream_utc.c",
              "SRCS:= deepstream_test5_app_main.c deepstream_utc.c court_filter.c jersey_log.c torso_objects.c "
              "courttrack_probe.cpp courttrack.cpp online_teams.cpp")
m = m.replace("CFLAGS+= $(shell pkg-config --cflags $(PKGS))",
              "CFLAGS+= $(shell pkg-config --cflags $(PKGS))\nCFLAGS+= -ffp-contract=off  # gym-mv3dt: reproducible court tracker\n"
              "CFLAGS+= -O2  # gym-mv3dt: NVIDIA's Makefile sets no optimisation; the court tracker at -O0 took 0.57 ms per "
              "frame and up to 42 ms in its slowest frames (replay), against 0.09 / 4 ms at -O2, with identical output")
mk.write_text(m)
EOF
( cd "$DST" && diff -u "$SRC/deepstream_test5_app_main.c" deepstream_test5_app_main.c > "$ROOT/apps/court_filter/test5_main.patch" || true )

docker run --rm -v "$DS:/ds" -w /ds/src/apps/sample_apps/deepstream-test5-court --entrypoint bash "$IMAGE" \
  -c "CV=\$(ls -d /usr/local/cuda-*.* | sed s/.*cuda-// | sort -V | tail -1); make CUDA_VER=\$CV -j8 >/tmp/build.log 2>&1 || { tail -40 /tmp/build.log; exit 1; }; chown $(id -u):$(id -g) deepstream-test5-court-app *.o ../deepstream-app/*.o ../../common/src/*.o ../../common/src/deepstream-yaml/*.o ../../../utils/nvds_msgapi/common_src/*.o 2>/dev/null || true"
docker run --rm -v "$ROOT/apps/jersey:/src" -w /src --entrypoint bash "$IMAGE" \
  -c "make -s CUDA_VER=\$(ls -d /usr/local/cuda-*.* | sed s/.*cuda-// | sort -V | tail -1) 2>&1 | grep -v Winfinite-recursion | grep -E 'error' ; chown $(id -u):$(id -g) libnvdsinfer_jersey.so"
docker run --rm -v "$ROOT/apps/yolox:/src" -w /src --entrypoint bash "$IMAGE" \
  -c "make -s CUDA_VER=\$(ls -d /usr/local/cuda-*.* | sed s/.*cuda-// | sort -V | tail -1) 2>&1 | grep -v Winfinite-recursion | grep -E 'error' ; chown $(id -u):$(id -g) libnvdsinfer_yolox.so"
mkdir -p "$ROOT/external/bin"
cp "$DST/deepstream-test5-court-app" "$ROOT/apps/jersey/libnvdsinfer_jersey.so" "$ROOT/apps/yolox/libnvdsinfer_yolox.so" \
  "$ROOT/external/bin/"
echo "built external/bin/deepstream-test5-court-app, external/bin/libnvdsinfer_jersey.so, external/bin/libnvdsinfer_yolox.so"
