"""Write the config for the court tracker probe (apps/courttrack/courttrack_probe.cpp, deepstream-test5-court-app).

  court <xmin> <xmax> <yhalf> <margin> <frame_w> <frame_h>     court lines (world metres) and the filter margin
  appearance <every> <min_box_h>                               torso colour every n-th frame, minimum box height
  seated <gap> <edge>                                          seated-person rule (tools/world_tracker.seated)
  camera <source> <name> <tracker index> <colour 0/1> <height 0/1> <bias_x> <bias_y> <12 x P>
  kafka <adapter library> <connection> <topic> <adapter config or ->
  ident_log <path>     CSV: frame, sensor, object id, court track, identity of every assigned detection
  input_log <path>     the tracker's inputs in the replay format (apps/courttrack/ct_replay, CTIN2)

Source ids follow the auto-configurator's sorted video order (as tools/court_filter_config); tracker camera indices
follow camera names (as tools/world_tracker, whose births sort cameras by name). The settings mirror the Python
reference: court filter margin 0.3 m (track_run), colour from good crops of FL, FR and NR every 3rd frame, heights from
FL and FR (tools/appearance), bias from camInfo courtTrackerBias_m (tools/calib_bias).

  python -m tools.courttrack_config <camInfo dir> <out file> [--kafka localhost;9092] [--topic courttrack]
"""

import argparse
from pathlib import Path

import yaml

from tools import CALIB_HEIGHT, CALIB_WIDTH, CAMERAS
from tools.appearance import COLOUR_CAMS, HEIGHT_CAMS, MIN_BOX_H
from tools.calib_bias import KEY
from tools.court import FT, HALF_FLAG_FT, HALF_LOUNGE_FT, WIDTH_FT
from tools.world_tracker import WTParams

KAFKA_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
LOGS = "/workspace/experiments"


def config_text(caminfo: Path, kafka: str | None, topic: str, margin: float = 0.3, every: int = 3) -> str:
    lines = [f"court {-HALF_FLAG_FT * FT:.4f} {HALF_LOUNGE_FT * FT:.4f} {WIDTH_FT * FT / 2:.4f} {margin} "
             f"{CALIB_WIDTH} {CALIB_HEIGHT}",
             f"appearance {every} {MIN_BOX_H}"]
    p = WTParams()
    if p.seated_gap is not None:
        lines.append(f"seated {p.seated_gap} {p.seated_edge}")
    index = {pos: i for i, pos in enumerate(sorted(pos for _, pos, _ in CAMERAS))}
    for sid, (cam_id, pos, _) in enumerate(CAMERAS):
        info = yaml.safe_load((caminfo / f"{cam_id}_{pos}.yml").read_text())
        if KEY not in info:
            raise SystemExit(f"{cam_id}_{pos}.yml has no {KEY}: measure it with tools/calib_bias.py --write")
        bx, by = info[KEY]
        P = info["projectionMatrix_3x4_w2p"]
        lines.append(f"camera {sid} {pos} {index[pos]} {int(pos in COLOUR_CAMS)} {int(pos in HEIGHT_CAMS)} "
                     f"{bx} {by} " + " ".join(f"{v:.10g}" for v in P))
    if kafka:
        lines.append(f"kafka {KAFKA_LIB} {kafka} {topic} -")
    lines += [f"ident_log {LOGS}/courttrack_identities.csv", f"input_log {LOGS}/courttrack_inputs.bin"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caminfo", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--kafka", default="localhost;9092", help="'' for no Kafka output")
    ap.add_argument("--topic", default="courttrack")
    a = ap.parse_args()
    a.out.write_text(config_text(a.caminfo, a.kafka or None, a.topic))


if __name__ == "__main__":
    main()
