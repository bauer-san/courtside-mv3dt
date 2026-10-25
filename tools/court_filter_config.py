"""Write the config for apps/court_filter (deepstream-test5-court-app).

Line 1: xmin xmax ymin ymax frame_height (court + margin, world metres). Then one line per source:
source_id followed by the 12 values of its projectionMatrix_3x4_w2p. Source ids follow the auto-configurator's
sorted video order (cam_00_NL=0, cam_01_FL=1, cam_02_NR=2, cam_03_FR=3). The matrices are only valid at
1920x1080: the streammux must run at that resolution.

  python -m tools.court_filter_config <camInfo dir> <out file> [--margin 0.75]
"""

import argparse
from pathlib import Path

import yaml

from tools import CALIB_HEIGHT, CAMERAS
from tools.court import FT, HALF_FLAG_FT, HALF_LOUNGE_FT, WIDTH_FT


def config_text(caminfo: Path, margin: float) -> str:
    lines = [f"{-HALF_FLAG_FT * FT - margin:.4f} {HALF_LOUNGE_FT * FT + margin:.4f} "
             f"{-WIDTH_FT * FT / 2 - margin:.4f} {WIDTH_FT * FT / 2 + margin:.4f} {CALIB_HEIGHT}"]
    for sid, (cam_id, pos, _) in enumerate(CAMERAS):
        p = yaml.safe_load((caminfo / f"{cam_id}_{pos}.yml").read_text())["projectionMatrix_3x4_w2p"]
        lines.append(f"{sid} " + " ".join(f"{v:.10g}" for v in p))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caminfo", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--margin", type=float, default=0.75)
    args = ap.parse_args()
    args.out.write_text(config_text(args.caminfo, args.margin))


if __name__ == "__main__":
    main()
