"""Render the game with the court tracker's player labels: 2x2 camera grid + top-down court, 1280x720 H.264.

Labels are the identity output of the default court tracker (tools/world_tracker.py): team + number, coloured by
team; refs grey; tracks without an identity white with their track number. At the labelled ground-truth moments
(gt/*) a strip shows how many labelled detections carry the right label, and wrong boxes are outlined in yellow.

  python -m tools.render_labels mount_r1 [--start 0] [--end 12600] [--out experiments/mount_r1/labelled_game.mp4]
"""

import argparse
import collections
import subprocess
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import load_boxes, run_dataset
from tools.court import court_lines
from tools.evaluate import X_MAX, X_MIN, Y_HALF
from tools.gt_eval import load_gt
from tools.world_tracker import WTParams, track_run

W, H = 1280, 720
CW, CH = 480, 270                       # one camera tile
GRID = [("NL", 0, 0), ("NR", 1, 0), ("FL", 0, 1), ("FR", 1, 1)]   # (cam, col, row): each end's cameras in a column
TEAM = {"red": (60, 60, 230), "maroon": (170, 70, 190), "ref": (170, 170, 170)}
UNKNOWN = (235, 235, 235)
BG, INK, MUTED = (24, 22, 20), (240, 240, 240), (150, 150, 150)
# court panel: long axis vertical, flag end at the bottom
PX0, PW, PY0, PH = 980, 280, 30, 660


def court_xy(x, y):
    s = min(PW / (2 * (Y_HALF + 1.0)), PH / (X_MAX - X_MIN + 2.0))
    cx, cy = PX0 + PW / 2, PY0 + PH / 2
    return int(cx - y * s), int(cy - x * s)


def colour(label: str):
    if label.startswith("t"):
        return UNKNOWN
    return TEAM.get(label.split(":")[0], UNKNOWN)


def text(label: str) -> str:
    if label.startswith("t"):
        return label
    team, num = label.split(":")
    return f"{team[0].upper()}{num}" if team in ("red", "maroon") else team


def tag(img, x, y, s, col, scale=0.4):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), col, -1)
    dark = sum(col) < 380
    cv2.putText(img, s, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, INK if dark else (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=12600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    out = args.out or run / "labelled_game.mp4"

    print("tracking ...", flush=True)
    T = track_run(run, WTParams(online_relink=True))
    di = track_run.last_det_ident
    label, pos = {}, collections.defaultdict(list)          # det key -> label; frame -> [(x, y, label)]
    for t in T:
        lab_at = {}
        for k in t.dets:
            k = (int(k[0]), str(k[1]), str(k[2]))
            label[k] = di.get(k) or f"t{t.tid}"
            lab_at.setdefault(k[0], label[k])
        for f, x, y, _ in t.out:
            if args.start <= f <= args.end:
                pos[f].append((x, y, lab_at.get(f, f"t{t.tid}"), t.tid))
    print("boxes ...", flush=True)
    boxes = load_boxes(run, 1)
    gt = {}
    for name in ("game_2to3", "game_4to5"):
        gt.update(load_gt(name, run))
    gt_frames = sorted({k[0] for k in gt})

    ds = run_dataset(run)
    caps = {pos_: cv2.VideoCapture(str((ds / "videos" / f"{cid}_{pos_}.mp4").resolve())) for cid, pos_, _ in CAMERAS}
    for c in caps.values():
        c.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    base = np.full((H, W, 3), BG, np.uint8)
    for _, pts in court_lines(60):
        p = np.array([court_xy(x, y) for x, y in pts], np.int32)
        cv2.polylines(base, [p], False, (90, 90, 90), 1, cv2.LINE_AA)
    top, bottom = court_xy(X_MAX, 0)[1], court_xy(X_MIN, 0)[1]
    cv2.putText(base, "flag end (NL+FR mount)", (PX0 + PW // 2 - 75, bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                MUTED, 1, cv2.LINE_AA)
    cv2.putText(base, "lounge end (NR+FL mount)", (PX0 + PW // 2 - 82, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                MUTED, 1, cv2.LINE_AA)
    cv2.putText(base, "top-down court", (PX0 + PW // 2 - 50, top - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1,
                cv2.LINE_AA)
    legend = [("red", TEAM["red"]), ("maroon", TEAM["maroon"]), ("ref", TEAM["ref"]), ("no identity yet", UNKNOWN)]
    x0 = 12
    for s, col in legend:
        cv2.rectangle(base, (x0, 600), (x0 + 12, 612), col, -1)
        cv2.putText(base, s, (x0 + 18, 611), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
        x0 += 30 + 9 * len(s)
    cv2.putText(base, f"{run.name}: court tracker identity output (team + jersey number). Cameras on their backboard "
                "mounts: NL+FR flag end, NR+FL lounge end.", (12, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1,
                cv2.LINE_AA)
    trail = collections.defaultdict(collections.deque)
    check = None                                             # (until frame, text)
    for f in range(args.start, args.end + 1):
        img = base.copy()
        is_gt = f in gt_frames
        wrong_keys, n_ok, n_all = set(), 0, 0
        if is_gt:
            for k, ident in gt.items():
                if k[0] == f:
                    n_all += 1
                    if label.get(k) == ident:
                        n_ok += 1
                    else:
                        wrong_keys.add(k)
            mm, ss = divmod(f // 30, 60)
            check = (f + 45, f"labelled moment {mm}:{ss:02d}: {n_ok} of {n_all} labels correct")
        for cam, col_, row_ in GRID:
            ok, frame = caps[cam].read()
            if not ok:
                continue
            s = CW / frame.shape[1]
            tile = cv2.resize(frame, (CW, CH), interpolation=cv2.INTER_AREA)
            for g, l, t, r, b in boxes.get((cam, f), []):
                k = (f, cam, str(g))
                lab = label.get(k)
                if lab is None:
                    continue
                c = colour(lab)
                p1, p2 = (int(l * s), int(t * s)), (int(r * s), int(b * s))
                cv2.rectangle(tile, p1, p2, c, 2 if not lab.startswith("t") else 1)
                if k in wrong_keys:
                    cv2.rectangle(tile, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3), (0, 230, 255), 2)
                tag(tile, p1[0], max(p1[1], 14), text(lab), c)
            cv2.putText(tile, cam, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
            y0, x0 = 30 + row_ * CH, 10 + col_ * CW
            img[y0:y0 + CH, x0:x0 + CW] = tile
        for x, y, lab, tid in pos.get(f, []):
            q = trail[tid]
            q.append(court_xy(x, y))
            while len(q) > 60:
                q.popleft()
        for tid in list(trail):
            if not any(p[3] == tid for p in pos.get(f, [])):
                if trail[tid]:
                    trail[tid].popleft()
                if not trail[tid]:
                    del trail[tid]
        for x, y, lab, tid in pos.get(f, []):
            c = colour(lab)
            q = trail.get(tid)
            if q and len(q) > 1:
                cv2.polylines(img, [np.array(q, np.int32)], False, tuple(int(v * 0.6) for v in c), 1, cv2.LINE_AA)
            p = court_xy(x, y)
            cv2.circle(img, p, 6, c, -1, cv2.LINE_AA)
            tag(img, p[0] + 7, p[1] + 4, text(lab), c, 0.35)
        mm, ss = divmod(f // 30, 60)
        cv2.putText(img, f"game clock {mm}:{ss:02d}  (frame {f})", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1,
                    cv2.LINE_AA)
        if check and f <= check[0]:
            cv2.rectangle(img, (10, 618), (960, 646), (0, 120, 160), -1)
            cv2.putText(img, check[1] + "  (wrong boxes outlined yellow on that frame)", (18, 637),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
        if f % 1800 == 0:
            print(f"frame {f}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    main()
