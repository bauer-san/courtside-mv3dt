"""Ball detection test: how often can a COCO detector (YOLOX, "sports ball") find the ball in our 30 fps cameras?

Each 1920 x 1080 frame is cut into six overlapping 640 x 640 tiles at full resolution (a whole frame scaled to 640 would
leave the ball ~5 px across); YOLOX runs on each tile (letterbox-free, BGR 0-255, as the official ONNX demo) and ball
boxes above --score are kept, merged across tiles by non-maximum suppression. Frames: every 3rd frame of the 2 s before
each labelled shot reached the rim (experiments/shots/shots_gt.json), and --sample random game frames for comparison.

  python -m tools.ball_detect ct3_r1 [--model yolox_l] [--score 0.1] -> experiments/shots/ball_<model>.json
  python -m tools.ball_detect ct3_r1 --summary                         # detection rates by camera and moment
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import run_dataset

BALL = 32                               # COCO class index of "sports ball"
TILES = [(x, y) for y in (0, 440) for x in (0, 640, 1280)]   # 640 x 640 tiles covering 1920 x 1080
SHOTS = ROOT / "experiments" / "shots"


def session(model: str):
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    return ort.InferenceSession(str(ROOT / "models" / "yolox" / f"{model}.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


_GRID = None


def decode(out: np.ndarray) -> np.ndarray:
    """YOLOX raw output (8400, 85) for a 640 x 640 input -> (8400, 85) with boxes as cx, cy, w, h in input pixels."""
    global _GRID
    if _GRID is None:
        grids, strides = [], []
        for s in (8, 16, 32):
            n = 640 // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        _GRID = (np.concatenate(grids), np.concatenate(strides))
    g, st = _GRID
    out = out.copy()
    out[:, :2] = (out[:, :2] + g) * st
    out[:, 2:4] = np.exp(out[:, 2:4]) * st
    return out


def detect(s, img: np.ndarray, score: float) -> list[tuple[float, float, float, float, float]]:
    """Ball boxes (x0, y0, x1, y1, score) in image pixels."""
    boxes = []
    for x, y in TILES:
        tile = img[y:y + 640, x:x + 640]
        inp = tile.transpose(2, 0, 1)[None].astype(np.float32)
        out = decode(s.run(None, {"images": inp})[0][0])
        sc = out[:, 4] * out[:, 5 + BALL]
        for i in np.where(sc >= score)[0]:
            cx, cy, w, h = out[i, :4]
            boxes.append((x + cx - w / 2, y + cy - h / 2, x + cx + w / 2, y + cy + h / 2, float(sc[i])))
    if not boxes:
        return []
    b = np.array(boxes)
    keep = cv2.dnn.NMSBoxes([[float(v[0]), float(v[1]), float(v[2] - v[0]), float(v[3] - v[1])] for v in b],
                            b[:, 4].tolist(), score, 0.4)
    return [tuple(map(float, b[i])) for i in np.array(keep).reshape(-1)]


def drop_static(rows: list[dict], cell: int = 20, share: float = 0.1) -> dict[str, int]:
    """Remove detections that sit in the same place (cell px) in more than `share` of a camera's frames (balls on a
    rack, lights, logos): in place; returns how many were removed per camera."""
    removed = {}
    for cam in ("NL", "FL", "NR", "FR"):
        cnt = {}
        for r in rows:
            for b in r["balls"].get(cam, []):
                k = (int((b[0] + b[2]) / 2 // cell), int((b[1] + b[3]) / 2 // cell))
                cnt[k] = cnt.get(k, 0) + 1
        n = sum(1 for r in rows if cam in r["balls"])
        bad = {k for k, v in cnt.items() if v > share * n}
        bad |= {(x + dx, y + dy) for x, y in bad for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        removed[cam] = 0
        for r in rows:
            keep = [b for b in r["balls"].get(cam, [])
                    if (int((b[0] + b[2]) / 2 // cell), int((b[1] + b[3]) / 2 // cell)) not in bad]
            removed[cam] += len(r["balls"].get(cam, [])) - len(keep)
            if cam in r["balls"]:
                r["balls"][cam] = keep
    return removed


def frames_to_check(sample: int, end: int, seed: int = 0) -> dict[int, str]:
    shots = json.loads((SHOTS / "shots_gt.json").read_text())
    out = {}
    for s in shots:
        for f in range(s["frame"] - 60, s["frame"] + 1, 3):
            out[f] = "before a shot"
    rng = random.Random(seed)
    while sum(1 for v in out.values() if v == "random") < sample:
        f = rng.randrange(0, end)
        out.setdefault(f, "random")
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--model", default="yolox_l", choices=("yolox_l", "yolox_m"))
    ap.add_argument("--score", type=float, default=0.1)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--end", type=int, default=12600)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_dir() else ROOT / "experiments" / a.run
    out_path = SHOTS / f"ball_{a.model}.json"
    if a.summary:
        res = json.loads(out_path.read_text())
        static = drop_static(res["frames"])
        print(f"static detections removed (same place in > 10 % of a camera's frames): {static}")
        print(f"| moment | camera | frames | ball found (score >= {a.score}) | ball found (score >= 0.3) |")
        print("|---|---|---|---|---|")
        for kind in ("before a shot", "random"):
            for cam in ("NL", "FL", "NR", "FR", "any"):
                rows = [r for r in res["frames"] if r["kind"] == kind]
                if cam == "any":
                    hit = [any(any(b[4] >= t for b in r["balls"].get(c, [])) for c in r["balls"]) for r in rows for t in (a.score,)]
                    hit3 = [any(any(b[4] >= 0.3 for b in r["balls"].get(c, [])) for c in r["balls"]) for r in rows]
                else:
                    hit = [any(b[4] >= a.score for b in r["balls"].get(cam, [])) for r in rows]
                    hit3 = [any(b[4] >= 0.3 for b in r["balls"].get(cam, [])) for r in rows]
                print(f"| {kind} | {cam} | {len(rows)} | {np.mean(hit):.0%} | {np.mean(hit3):.0%} |")
        print(f"\ntime per frame (4 cameras x 6 tiles): {res['ms_per_frame']:.0f} ms ({a.model}, RTX 3060 Ti)")
        return
    s = session(a.model)
    todo = frames_to_check(a.sample, a.end)
    ds = run_dataset(run)
    rows = {f: {"frame": f, "kind": k, "balls": {}} for f, k in todo.items()}
    t0, n = time.time(), 0
    for cid, cam, _ in CAMERAS:
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cid}_{cam}.mp4").resolve()))
        cur = 0
        for f in todo:
            if f < cur:
                continue
            if f - cur > 90:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                cur = f
            while cur < f:
                cap.grab()
                cur += 1
            ok, img = cap.read()
            cur += 1
            if not ok:
                continue
            rows[f]["balls"][cam] = detect(s, img, a.score)
            n += 1
        print(cam, "done", flush=True)
    ms = (time.time() - t0) * 1000 / max(n, 1) * 4
    out_path.write_text(json.dumps({"model": a.model, "score": a.score, "ms_per_frame": ms, "frames": list(rows.values())}))
    print(out_path)


if __name__ == "__main__":
    main()
