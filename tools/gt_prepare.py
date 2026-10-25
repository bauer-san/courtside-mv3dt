"""Small identity ground truth: annotation tasks for the court-side players.

For `--moments` frames (every `--step` frames from `--start`), every person the deployable court tracker
(online relink, team colour + jersey numbers) holds at that frame becomes one *group*: the camera crops of all
detections assigned to it at that frame. Court-filtered detections the tracker did not assign become
single-crop groups. The annotator gives each group an identity (and may override single crops); labels attach
to detection keys (frame, cam, gid), so any tracker that consumes the same detections can be scored.

Output in gt/<name>/: tasks.json (moments, groups, crops, roster, prefilled guesses) and one crop sprite per
moment (sprites/<frame>.jpg). The crops are cut from the dataset videos (frame = video frame index).

  python -m tools.gt_prepare <run> [--name game_2to3] [--start 3600] [--step 60] [--moments 30]
"""

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import load_boxes, run_dataset
from tools.court import court_lines
from tools.evaluate import inside
from tools.stitch import Params as StitchParams, ground_filter, load_cameras, load_detections
from tools.world_tracker import WTParams, track_run

CROP_H = 220          # px, crop height in the sprite
SPRITE_W = 2200       # px, sprite row width
JPEG_Q = 80
CAM_ORDER = {pos: i for i, (_, pos, _) in enumerate(CAMERAS)}


def roster(tracks: list, p: WTParams, max_frame: int) -> dict:
    """team -> numbers seen on at least two tracks of that team (or one track with >= 10 reads)."""
    tally = collections.defaultdict(collections.Counter)
    for t in tracks:
        team, num = t.team(p), t.number(p)
        if team in ("red", "maroon") and num and t.out[0][0] <= max_frame:
            tally[team][num] += 1 + (sum(t.nvotes.values()) >= 10)
    return {team: sorted((n for n, k in c.items() if k >= 2), key=int) for team, c in sorted(tally.items())}


def guess(t, p: WTParams) -> str:
    team, num = t.team(p), t.number(p)
    if team == "ref":
        return "ref ?"
    if team in ("red", "maroon"):
        return f"{team} {num}" if num else f"{team} ?"
    return f"? {num}" if num else ""


def pack(crops: list) -> tuple[np.ndarray, list]:
    """Shelf-pack crops (same height) into rows; returns the sprite and (x, y, w, h) per crop."""
    pos, x, y = [], 0, 0
    for c in crops:
        w = c.shape[1]
        if x and x + w > SPRITE_W:
            x, y = 0, y + CROP_H
        pos.append((x, y, w, CROP_H))
        x += w
    H = y + CROP_H
    W = max(px + w for px, _, w, _ in pos)
    img = np.full((H, W, 3), 255, np.uint8)
    for c, (px, py, w, h) in zip(crops, pos):
        img[py:py + h, px:px + w] = c
    return img, pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--name", default="game_2to3")
    ap.add_argument("--start", type=int, default=3600)
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--moments", type=int, default=30)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    frames = [args.start + i * args.step for i in range(args.moments)]
    out = ROOT / "gt" / args.name
    (out / "sprites").mkdir(parents=True, exist_ok=True)

    p = WTParams(online_relink=True)
    tracks = track_run(run, p)
    d = load_detections(run / "kafka.jsonl.gz")
    keep = ground_filter(d, load_cameras(run / "camInfo_used"), inside, StitchParams(margin=0.3))
    boxes = load_boxes(run, args.step)
    box = {(cam, fr, str(g)): (l, t, r, b) for (cam, fr), v in boxes.items() for g, l, t, r, b in v}

    fset = set(frames)
    groups = collections.defaultdict(list)          # frame -> [group]
    assigned = set()
    for t in tracks:
        here = collections.defaultdict(list)
        for k in t.dets:
            if k[0] in fset:
                here[k[0]].append(k)
        pos = {f: (x, y) for f, x, y, _ in t.out if f in fset}
        for f, keys in here.items():
            assigned.update(keys)
            groups[f].append(dict(track=t.tid, guess=guess(t, p), xy=pos.get(f), keys=keys))
    for fr, cam, gid, x, y in zip(d["frame"][keep], d["cam"][keep], d["gid"][keep], d["x"][keep], d["y"][keep]):
        k = (int(fr), str(cam), str(gid))
        if k[0] in fset and k not in assigned:
            groups[k[0]].append(dict(track=None, guess="", xy=(float(x), float(y)), keys=[k]))

    # cut the crops: (cam, frame) -> image, one pass per video
    need = collections.defaultdict(set)
    for f, gs in groups.items():
        for g in gs:
            for _, cam, _ in g["keys"]:
                need[cam].add(f)
    images = {}
    ds = run_dataset(run)
    for cam_id, pos_name, _ in CAMERAS:
        want = sorted(need[pos_name])
        if not want:
            continue
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos_name}.mp4").resolve()))
        for f in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                images[(pos_name, f)] = img
        cap.release()

    moments = []
    n_groups = n_crops = 0
    for f in frames:
        gs = sorted(groups.get(f, []), key=lambda g: (g["xy"] or (99, 0))[0])
        crops, meta = [], []
        for gi, g in enumerate(gs):
            g["id"] = f"{f}_{gi}"
            g["crops"] = []
            for _, cam, gid in sorted(g["keys"], key=lambda k: CAM_ORDER[k[1]]):
                b = box.get((cam, f, gid))
                img = images.get((cam, f))
                if b is None or img is None:
                    continue
                l, t_, r, b_ = (int(round(v)) for v in b)
                py, px = int(0.08 * (b_ - t_)), int(0.35 * (r - l))      # some context; the box is outlined
                y0, x0 = max(0, t_ - py), max(0, l - px)
                c = img[y0:min(img.shape[0], b_ + py), x0:min(img.shape[1], r + px)].copy()
                if c.size == 0:
                    continue
                lw = max(1, round(c.shape[0] / CROP_H))
                cv2.rectangle(c, (l - x0, t_ - y0), (r - x0 - 1, b_ - y0 - 1), (0, 220, 255), 2 * lw)
                s = CROP_H / c.shape[0]
                crops.append(cv2.resize(c, (max(8, int(c.shape[1] * s)), CROP_H), interpolation=cv2.INTER_AREA))
                meta.append((g, dict(cam=cam, gid=gid, h=b_ - t_)))
        if not crops:
            continue
        sprite, rects = pack(crops)
        cv2.imwrite(str(out / "sprites" / f"{f}.jpg"), sprite, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        for (g, c), r in zip(meta, rects):
            c["r"] = list(r)
            g["crops"].append(c)
        gs = [g for g in gs if g["crops"]]
        n_groups += len(gs)
        n_crops += len(crops)
        moments.append(dict(frame=f, t=round(f / 30, 1), sprite=f"sprites/{f}.jpg", size=list(sprite.shape[1::-1]),
                            groups=[dict(id=g["id"], track=g["track"], guess=g["guess"],
                                         xy=[round(v, 2) for v in g["xy"]] if g["xy"] else None,
                                         crops=g["crops"]) for g in gs]))

    tasks = dict(run=run.name, name=args.name, frames=frames, roster=roster(tracks, p, 12600),
                 tracker="court tracker, online relink, team colour + DeepStream numbers", moments=moments,
                 court=[np.round(pts, 2).tolist() for _, pts in court_lines(40)])
    (out / "tasks.json").write_text(json.dumps(tasks, separators=(",", ":")))
    print(f"{out}: {len(moments)} moments, {n_groups} groups, {n_crops} crops; roster {tasks['roster']}")


if __name__ == "__main__":
    main()
