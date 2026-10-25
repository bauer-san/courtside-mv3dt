"""Candidate shot moments from rim motion (tools/rim_motion.py), with a short clip of each for labelling.

A candidate is a burst of motion inside a rim (NL: left-end rim, NR: right-end rim): the 5-frame mean of the changed
pixel share above --low, grouped within 2 s, kept if its peak reaches --thr. Frames where the whole rim window changes at
once (> 0.5: decoder or exposure glitches) and the first 2 s are ignored. Each clip is 3.5 s (2 s before the peak to
1.5 s after): the rim seen from its backboard camera on the left, the wide view from the far camera that faces that end
(FL for the left end, FR for the right end) on the right.

  python -m tools.shot_candidates [--thr 0.03] -> experiments/shots/candidates.json, experiments/shots/clips/cNN.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from tools import DATASETS, ROOT

VIDEO = {"NL": "cam_00_NL.mp4", "FL": "cam_01_FL.mp4", "NR": "cam_02_NR.mp4", "FR": "cam_03_FR.mp4"}
WIDE = {"NL": "FL", "NR": "FR"}
END = {"NL": "left", "NR": "right"}
PRE, POST = 60, 45


def find(sig: np.ndarray, thr: float, low: float) -> list[tuple[int, float]]:
    s = sig.copy()
    s[s > 0.5] = 0
    s[:60] = 0
    sm = np.convolve(s, np.ones(5) / 5, "same")
    ev = []
    for i in np.where(sm > low)[0]:
        if ev and i - ev[-1][1] <= 60:
            ev[-1][1] = i
        else:
            ev.append([i, i])
    out = []
    for a, b in ev:
        pk = a + int(np.argmax(sm[a:b + 1]))
        if sm[pk] >= thr:
            out.append((pk, float(sm[pk])))
    return out


def clip(cam: str, peak: int, box, out: Path) -> None:
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    caps = {c: cv2.VideoCapture(str((DATASETS / "gym_full" / "videos" / VIDEO[c]).resolve())) for c in (cam, WIDE[cam])}
    for c in caps.values():
        c.set(cv2.CAP_PROP_POS_FRAMES, max(0, peak - PRE))
    w, h = 1000, 360
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in range(max(0, peak - PRE), peak + POST):
        ok1, a = caps[cam].read()
        ok2, b = caps[WIDE[cam]].read()
        if not (ok1 and ok2):
            break
        H, W = a.shape[:2]
        ya, xa = max(0, cy - side // 2), max(0, cx - side // 2)
        rim = cv2.resize(a[ya:min(H, ya + side), xa:min(W, xa + side)], (360, 360))
        wide = cv2.resize(b, (640, 360))
        frame = np.hstack([rim, wide])
        mm, ss = divmod(f // 30, 60)
        cv2.putText(frame, f"{cam} rim", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{WIDE[cam]}  {mm}:{ss:02d}  ({f - peak:+d})", (368, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thr", type=float, default=0.03)
    ap.add_argument("--low", type=float, default=0.03)
    ap.add_argument("--motion", type=Path, default=ROOT / "experiments" / "shots" / "rim_motion.npz")
    a = ap.parse_args()
    z = np.load(a.motion)
    cands = sorted(((cam, pk, v) for cam in ("NL", "NR") for pk, v in find(z[cam], a.thr, min(a.low, a.thr))),
                   key=lambda c: c[1])
    outdir = a.motion.parent / "clips"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (cam, pk, v) in enumerate(cands, 1):
        cid = f"c{i:02d}"
        clip(cam, pk, z[cam + "_box"], outdir / f"{cid}.mp4")
        mm, ss = divmod(pk // 30, 60)
        rows.append({"id": cid, "camera": cam, "end": END[cam], "frame": int(pk), "time": f"{mm}:{ss:02d}",
                     "peak": round(v, 3)})
        print(cid, cam, rows[-1]["time"], rows[-1]["peak"], flush=True)
    (a.motion.parent / "candidates.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
