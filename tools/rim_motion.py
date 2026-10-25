"""Motion inside each rim, seen from the backboard cameras (NL: flag-end rim, NR: lounge-end rim).

NL and NR sit on top of the backboards looking down, so each sees its own rim at a fixed place in the image. Per frame,
the share of pixels inside the projected rim (plus a margin) that changed since the previous frame; a ball arriving at
the rim, and the net moving, make it jump. Candidate shot moments are the peaks of that signal.

  python -m tools.rim_motion [--end 12600] [--out experiments/shots/rim_motion.npz]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from tools import DATASETS, ROOT
from tools.court import FT, HALF_FLAG_FT, HALF_LOUNGE_FT
from tools.stitch import load_cameras

RIM_R = 0.2286          # m (18 in diameter)
RIMS = {"NL": np.array([-(HALF_FLAG_FT - 5.25) * FT, 0.0, 10 * FT]),
        "NR": np.array([(HALF_LOUNGE_FT - 5.25) * FT, 0.0, 10 * FT])}
VIDEO = {"NL": "cam_00_NL.mp4", "NR": "cam_02_NR.mp4"}


def rim_mask(P: np.ndarray, X: np.ndarray, shape, margin: float = 1.4) -> np.ndarray:
    """Pixels inside the projected rim circle, enlarged by `margin`."""
    th = np.linspace(0, 2 * np.pi, 60)
    pts = np.c_[X[0] + margin * RIM_R * np.cos(th), X[1] + margin * RIM_R * np.sin(th), np.full(60, X[2]), np.ones(60)] @ P.T
    poly = (pts[:, :2] / pts[:, 2:]).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", type=int, default=12600)
    ap.add_argument("--diff", type=int, default=25, help="grey-level change counted as motion")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "shots" / "rim_motion.npz")
    a = ap.parse_args()
    cams = load_cameras(DATASETS / "courtside" / "camInfo")
    out = {}
    for cam in ("NL", "NR"):
        cap = cv2.VideoCapture(str((DATASETS / "courtside_full" / "videos" / VIDEO[cam]).resolve()))
        ok, prev = cap.read()
        mask = rim_mask(cams[cam], RIMS[cam], prev.shape[:2])
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = mask[y0:y1, x0:x1]
        prev = cv2.cvtColor(prev[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        sig = [0.0]
        for f in range(1, a.end + 1):
            ok, img = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            sig.append(float((cv2.absdiff(g, prev)[m] > a.diff).mean()))
            prev = g
        out[cam] = np.array(sig, np.float32)
        out[cam + "_box"] = np.array([x0, y0, x1, y1])
        print(cam, "frames", len(sig), "box", (x0, y0, x1, y1), "median", np.median(sig), "p99", np.percentile(sig, 99), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(a.out)


if __name__ == "__main__":
    main()
