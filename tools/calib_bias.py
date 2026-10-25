"""Measure each camera's floor-position bias for the court tracker and store it with the calibration.

The bias is a constant offset of a camera's MV3DT foot points relative to the other cameras (mainly NL and NR, about
0.18 m along the court, in opposite directions), estimated from detections of the same person in two cameras
(world_tracker.estimate_bias, offsets summing to zero). It is stable across runs, so it is measured once at
installation and stored in each camera's camInfo file as `courtTrackerBias_m: [x, y]`, which the court tracker
subtracts before tracking (the DeepStream probe and track_run(correct_bias="stored")). Re-measure whenever the
calibration changes.

  python -m tools.calib_bias mount_r1 mount_r2 mount_r3            # per-run and pooled values
  python -m tools.calib_bias mount_r1 mount_r2 mount_r3 --write    # store the pooled values in camInfo
"""

import argparse
import re
from pathlib import Path

import yaml

from tools import CAMERAS, DATASETS, ROOT

CAMINFO = DATASETS / "gym" / "camInfo"
KEY = "courtTrackerBias_m"


def run_by_frame(run: Path) -> dict:
    from tools.evaluate import inside
    from tools.stitch import Params as StitchParams
    from tools.world_tracker import _by_frame, ground_filter, load_cameras, load_detections
    d = load_detections(run / "kafka.jsonl.gz")
    keep = ground_filter(d, load_cameras(run / "camInfo_used"), inside, StitchParams(margin=0.3))
    return _by_frame(d, keep)[0]


def stored(caminfo: Path = CAMINFO) -> dict[str, tuple[float, float]]:
    """Camera position -> stored bias (empty if not measured)."""
    out = {}
    for _, pos, _ in CAMERAS:
        f = next(caminfo.glob(f"*_{pos}.yml"))
        v = yaml.safe_load(f.read_text()).get(KEY)
        if v is not None:
            out[pos] = (float(v[0]), float(v[1]))
    return out


def write(bias: dict, runs: list[str], caminfo: Path = CAMINFO) -> None:
    for _, pos, _ in CAMERAS:
        f = next(caminfo.glob(f"*_{pos}.yml"))
        text = re.sub(rf"\n# Court tracker position bias.*?\n{KEY}: .*?\n", "\n", f.read_text(), flags=re.S)
        bx, by = bias[pos]
        text = text.rstrip("\n") + (
            f"\n\n# Court tracker position bias (m): constant offset of this camera's foot points, subtracted before\n"
            f"# court tracking. Measured with tools/calib_bias.py on {', '.join(runs)}.\n"
            f"# Re-measure whenever the calibration above changes.\n"
            f"{KEY}: [{bx:.4f}, {by:.4f}]\n")
        f.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    from tools.world_tracker import estimate_bias
    pooled = {}
    for i, r in enumerate(a.runs):
        bf = run_by_frame(ROOT / "experiments" / r)
        b = estimate_bias(bf)
        print(f"{r}: " + "  ".join(f"{c} ({v[0]:+.3f}, {v[1]:+.3f})" for c, v in sorted(b.items())))
        pooled.update({i * 10 ** 7 + f: v for f, v in bf.items()})
    b = estimate_bias(pooled)
    print("pooled: " + "  ".join(f"{c} ({v[0]:+.3f}, {v[1]:+.3f})" for c, v in sorted(b.items())))
    if a.write:
        write(b, a.runs)
        print(f"written to {CAMINFO}/*.yml ({KEY})")


if __name__ == "__main__":
    main()
