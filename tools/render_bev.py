"""Render a BEV video of tracks (tracks_world.csv or tracks_clean.csv) over layout_v2.png.

  python -m tools.render_bev experiments/<run>/tracks_world.csv out.mp4 [--start 0 --seconds 20 --fps 15]
Each track: dot at its current position, ID label, 2 s trail. Colours are stable per track ID.
"""

import argparse
import collections
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tools import INPUTS
from tools.court import world_to_map

FPS = 30


def load(path: Path) -> dict[int, list]:
    tracks = collections.defaultdict(list)
    with path.open() as f:
        next(f)
        for line in f:
            t, fr, x, y, _ = line.strip().split(",")
            tracks[int(t)].append((int(fr), float(x), float(y)))
    return tracks


def colour(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 7919 + 13)
    return tuple(int(v) for v in rng.integers(40, 230, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    tracks = load(args.tracks)
    by_frame = collections.defaultdict(list)
    for tid, pts in tracks.items():
        for fr, x, y in pts:
            by_frame[fr].append((tid, x, y))
    first = min(by_frame)
    f0 = first + int(args.start * FPS)
    f1 = f0 + int(args.seconds * FPS)
    step = max(1, FPS // args.fps)
    base = cv2.imread(str(INPUTS / "layout_v2.png"))
    hist = collections.defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for fr in range(f0, f1):
            for tid, x, y in by_frame.get(fr, []):
                hist[tid].append((fr, x, y))
            if (fr - f0) % step:
                continue
            img = base.copy()
            alive = 0
            for tid, h in hist.items():
                recent = [p for p in h if fr - 2 * FPS <= p[0] <= fr]
                if not recent:
                    continue
                pts = world_to_map(np.array([[p[1], p[2]] for p in recent])).astype(np.int32)
                cv2.polylines(img, [pts], False, colour(tid), 2, cv2.LINE_AA)
                if fr - recent[-1][0] <= 2:
                    alive += 1
                    u, v = pts[-1]
                    cv2.circle(img, (int(u), int(v)), 8, colour(tid), -1, cv2.LINE_AA)
                    cv2.putText(img, str(tid), (int(u) + 9, int(v) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(img, f"{args.title}  t={(fr - first) / FPS:5.1f}s  tracks visible: {alive}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imwrite(f"{tmp}/{n:05d}.png", img)
            n += 1
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps),
                        "-i", f"{tmp}/%05d.png", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out)], check=True)
    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
