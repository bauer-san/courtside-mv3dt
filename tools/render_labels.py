"""Render the game with the court tracker's player labels: 2x2 camera grid + top-down court, 1280x720 H.264.

Labels are the identity output of the default court tracker (tools/world_tracker.py): team + number, coloured by
team; refs grey; tracks without an identity white with their track number. At the labelled ground-truth moments
(gt/*) a strip shows how many labelled detections carry the right label, and wrong boxes are outlined in yellow.

  python -m tools.render_labels mount_r1 [--start 0] [--end GAME_END] [--out experiments/mount_r1/labelled_game.mp4]
  python -m tools.render_labels ct3_r1 --live     # what a COURTTRACK=1 run sent: per-frame Kafka messages

--sanitize pixelates what could identify people or the venue: everything whose floor projection (calibration) lies
outside the court lines (crowd, walls, signs), the centre logo and the lanes (floor lettering), and every person's
head (top of each detection box). --still N writes only frame N as an image (--out x.jpg).

--live draws the DeepStream run's own output instead of re-running the tracker: positions and labels from the
per-frame messages (courttrack.jsonl.gz), detection identities from courttrack_identities.csv. Boxes that are not
in that frame's message (dropped seated spectators, unconfirmed tracks) are thin grey. The automatic team model names
the teams A and B; their colours follow the labelled team they match (for display and the label check only).
"""

import argparse
import collections
import csv
import gzip
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, GAME_END, ROOT
from tools.appearance import load_boxes, run_dataset
from tools.court import court_lines
from tools.evaluate import X_MAX, X_MIN, Y_HALF
from tools.gt_eval import load_gt
from tools.world_tracker import WTParams, track_default, track_run

W, H = 1280, 720
CW, CH = 480, 270                       # one camera tile
GRID = [("NL", 0, 0), ("NR", 1, 0), ("FL", 0, 1), ("FR", 1, 1)]   # (cam, col, row): each end's cameras in a column
TEAM = {"red": (60, 60, 230), "maroon": (170, 70, 190), "ref": (170, 170, 170)}
UNKNOWN = (235, 235, 235)
DIM = (110, 110, 110)
GT_NAME = {}                            # live team (A/B) -> labelled team name, for the label check
BG, INK, MUTED = (24, 22, 20), (240, 240, 240), (150, 150, 150)
# court panel: long axis vertical, left end at the bottom
PX0, PW, PY0, PH = 980, 280, 30, 660
Z = 1.0                                 # layout scale (--scale 1.5: 1920x1080)


def z(v: float) -> int:
    return int(round(v * Z))


def set_scale(scale: float) -> None:
    global Z, W, H, CW, CH, PX0, PW, PY0, PH
    Z = scale
    W, H, CW, CH = z(1280), z(720), z(480), z(270)
    W, H = W + W % 2, H + H % 2          # H.264 needs even sizes
    PX0, PW, PY0, PH = z(980), z(280), z(30), z(660)


def court_xy(x, y):
    s = min(PW / (2 * (Y_HALF + 1.0)), PH / (X_MAX - X_MIN + 2.0))
    cx, cy = PX0 + PW / 2, PY0 + PH / 2
    return int(cx - y * s), int(cy - x * s)


def colour(label: str):
    if label.startswith("t"):
        return UNKNOWN
    return TEAM.get(label.split(":")[0], UNKNOWN)


def floor_masks(cams: dict, w: int = 1920, h: int = 1080) -> dict:
    """Per camera: pixels to pixelate for --sanitize (outside the court + 0.4 m, centre logo disc, lanes)."""
    from tools.court import FT
    out = {}
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    for cam, P in cams.items():
        H = P[:, [0, 1, 3]]
        X = np.linalg.solve(H, np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))
        x, y = X[0] / X[2], X[1] / X[2]
        front = (P @ np.stack([x, y, np.zeros_like(x), np.ones_like(x)]))[2] > 0
        m = 0.4
        outside = ~front | (x < X_MIN - m) | (x > X_MAX + m) | (np.abs(y) > Y_HALF + m)
        logo = np.hypot(x, y) < 4.0
        lane_w, lane_l = 6 * FT + 0.15, 19 * FT + 0.15
        lanes = (np.abs(y) < lane_w) & ((x < X_MIN + lane_l) | (x > X_MAX - lane_l))
        out[cam] = (outside | logo | lanes).reshape(h, w)
    return out


_FACE = None


def faces(img: np.ndarray, thr: float = 0.15) -> list[tuple[int, int, int, int]]:
    """Face boxes from PeopleNet's face class (DetectNet_v2 output: stride 16, bbox norm 35, offset 0.5), on four
    full-resolution 960 x 544 tiles so that small faces are not lost."""
    global _FACE
    if _FACE is None:
        import onnxruntime as ort
        ort.preload_dlls()
        ort.set_default_logger_severity(3)
        _FACE = ort.InferenceSession(str(ROOT / "external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/"
                                         "models/PeopleNet2.6.3/resnet34_peoplenet.onnx"),
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    gx = (np.arange(60) * 16 + 0.5) / 35.0
    gy = (np.arange(34) * 16 + 0.5) / 35.0
    out = []
    H, W = img.shape[:2]
    for y0 in (0, H - 544):
        for x0 in (0, W - 960):
            tile = cv2.cvtColor(img[y0:y0 + 544, x0:x0 + 960], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            cov, bb = _FACE.run(None, {"input_1:0": tile.transpose(2, 0, 1)[None]})
            c, b = cov[0, 2], bb[0, 8:12]          # class 2 = Face
            for j, i in zip(*np.where(c >= thr)):
                x1 = (b[0, j, i] - gx[i]) * -35.0
                y1 = (b[1, j, i] - gy[j]) * -35.0
                x2 = (b[2, j, i] + gx[i]) * 35.0
                y2 = (b[3, j, i] + gy[j]) * 35.0
                out.append((x0 + x1, y0 + y1, x0 + x2, y0 + y2))
    return [tuple(map(int, f)) for f in out]


def pixelate(img: np.ndarray, block: int = 18) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def sanitize(img: np.ndarray, mask: np.ndarray, people: list) -> np.ndarray:
    """Pixelate `mask` except people's bodies, and every person's head (top 24 % of the box, widened)."""
    m = mask.copy()
    heads = np.zeros_like(m)
    for _, l, t, r, b in people:
        l, t, r, b = int(l), int(t), int(r), int(b)
        hh = int(0.24 * (b - t))
        m[max(0, t + hh):max(0, b), max(0, l):max(0, r)] = False
        heads[max(0, t - hh // 3):max(0, t + hh), max(0, l - (r - l) // 6):max(0, r + (r - l) // 6)] = True
    m |= heads
    for x1, y1, x2, y2 in faces(img):                 # faces, whether or not the person has a box
        w, h = x2 - x1, y2 - y1
        m[max(0, int(y1 - 0.3 * h)):max(0, int(y2 + 0.3 * h)), max(0, int(x1 - 0.3 * w)):max(0, int(x2 + 0.3 * w))] = True
    px = pixelate(img)
    img = img.copy()
    img[m] = px[m]
    return img


def text(label: str) -> str:
    if label.startswith("t"):
        return label
    team, num = label.split(":")
    if team in ("A", "B"):
        return f"{team}{num}"
    return f"{team[0].upper()}{num}" if team in ("red", "maroon") else team


def live_output(run: Path, start: int, end: int):
    """Labels, court positions and output detections of a COURTTRACK=1 run."""
    label = {}
    with open(run / "courttrack_identities.csv") as f:
        for r in csv.DictReader(f):
            k = (int(r["frame"]), r["sensor"], r["object_id"])
            label[k] = r["identity"] or f"t{r['track']}"
    pos, out_keys = collections.defaultdict(list), set()
    with gzip.open(run / "courttrack.jsonl.gz", "rt") as f:
        for line in f:
            m = json.loads(line)
            fr = m["frame"]
            if not start <= fr <= end:
                continue
            for p in m["players"]:
                lab = p["identity"] or f"t{p['track']}"
                pos[fr].append((p["x"], p["y"], lab, p["track"]))
                for d in p["detections"]:
                    out_keys.add((fr, d["sensor"], d["objectId"]))
    return label, pos, out_keys


def tag(img, x, y, s, col, scale=0.4):
    scale *= Z
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - z(4)), (x + tw + z(4), y), col, -1)
    dark = sum(col) < 380
    cv2.putText(img, s, (x + z(2), y - z(3)), cv2.FONT_HERSHEY_SIMPLEX, scale, INK if dark else (20, 20, 20), 1,
                cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=GAME_END)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--live", action="store_true", help="draw the run's live DeepStream output (COURTTRACK=1 runs)")
    ap.add_argument("--scale", type=float, default=1.0, help="layout scale: 1 = 1280x720, 1.5 = 1920x1080")
    ap.add_argument("--sanitize", action="store_true", help="pixelate faces, the crowd, signs and floor lettering")
    ap.add_argument("--still", type=int, help="write only this frame, as an image (--out x.jpg)")
    args = ap.parse_args()
    set_scale(args.scale)
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    if args.still is not None:
        args.start = args.end = args.still
    out = args.out or run / ("labelled_game_live.mp4" if args.live else "labelled_game.mp4")
    masks = None
    if args.sanitize:
        from tools.stitch import load_cameras
        masks = floor_masks(load_cameras(run / "camInfo_used"))
    out_keys = None
    if args.live:
        label, pos, out_keys = live_output(run, args.start, args.end)
    else:
        print("tracking ...", flush=True)
        T = track_default(run)
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
    if args.live:   # which labelled team each automatic team (A/B) corresponds to: majority over the labelled moments
        all_keys = out_keys if args.still is None else live_output(run, 0, 10 ** 9)[2]
        votes = collections.Counter((label[k].split(":")[0], v.split(":")[0]) for k, v in gt.items()
                                    if k in all_keys and ":" in label.get(k, ""))
        for team in ("A", "B"):
            best = max((n, g) for (t_, g), n in votes.items() if t_ == team) if any(t_ == team for t_, _ in votes) else None
            if best:
                GT_NAME[team] = best[1]
                TEAM[team] = TEAM.get(best[1], UNKNOWN)
    shown = lambda lab: (GT_NAME.get(lab.split(":")[0], lab.split(":")[0]) + ":" + lab.split(":")[1]) if ":" in lab else lab

    ds = run_dataset(run)
    caps = {pos_: cv2.VideoCapture(str((ds / "videos" / f"{cid}_{pos_}.mp4").resolve())) for cid, pos_, _ in CAMERAS}
    for c in caps.values():
        c.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ff = None if args.still is not None else subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    base = np.full((H, W, 3), BG, np.uint8)
    for _, pts in court_lines(60):
        p = np.array([court_xy(x, y) for x, y in pts], np.int32)
        cv2.polylines(base, [p], False, (90, 90, 90), 1, cv2.LINE_AA)
    top, bottom = court_xy(X_MAX, 0)[1], court_xy(X_MIN, 0)[1]
    cv2.putText(base, "left end (NL+FR mount)", (PX0 + PW // 2 - z(75), bottom + z(22)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * Z,
                MUTED, 1, cv2.LINE_AA)
    cv2.putText(base, "right end (NR+FL mount)", (PX0 + PW // 2 - z(82), top - z(12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * Z,
                MUTED, 1, cv2.LINE_AA)
    cv2.putText(base, "top-down court", (PX0 + PW // 2 - z(50), top - z(34)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * Z, INK, 1,
                cv2.LINE_AA)
    if args.live:
        legend = [(f"team {t} ({GT_NAME.get(t, '?')} shirts)", TEAM.get(t, UNKNOWN)) for t in ("A", "B")] + \
                 [("no identity yet", UNKNOWN), ("not in the output", DIM)]
    else:
        legend = [("red", TEAM["red"]), ("maroon", TEAM["maroon"]), ("ref", TEAM["ref"]), ("no identity yet", UNKNOWN)]
    x0 = z(12)
    for s, col in legend:
        cv2.rectangle(base, (x0, z(600)), (x0 + z(12), z(612)), col, -1)
        cv2.putText(base, s, (x0 + z(18), z(611)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * Z, INK, 1, cv2.LINE_AA)
        x0 += z(30 + 9 * len(s))
    caption = (f"{run.name}: live DeepStream output (per-frame Kafka messages): automatic team + jersey number." if args.live
               else f"{run.name}: court tracker identity output (team + jersey number).")
    cv2.putText(base, caption + " Cameras on their backboard mounts: NL+FR left end, NR+FL right end.", (z(12), z(690)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * Z, MUTED, 1, cv2.LINE_AA)
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
                    if shown(label.get(k, "")) == ident and (out_keys is None or k in out_keys):
                        n_ok += 1
                    else:
                        wrong_keys.add(k)
            mm, ss = divmod(f // 30, 60)
            check = (f + 45, f"labelled moment {mm}:{ss:02d}: {n_ok} of {n_all} labels correct")
        for cam, col_, row_ in GRID:
            ok, frame = caps[cam].read()
            if not ok:
                continue
            if masks is not None:
                frame = sanitize(frame, masks[cam], boxes.get((cam, f), []))
            s = CW / frame.shape[1]
            tile = cv2.resize(frame, (CW, CH), interpolation=cv2.INTER_AREA)
            for g, l, t, r, b in boxes.get((cam, f), []):
                k = (f, cam, str(g))
                lab = label.get(k)
                if out_keys is not None and k not in out_keys:
                    cv2.rectangle(tile, (int(l * s), int(t * s)), (int(r * s), int(b * s)), DIM, 1)
                    continue
                if lab is None:
                    continue
                c = colour(lab)
                p1, p2 = (int(l * s), int(t * s)), (int(r * s), int(b * s))
                cv2.rectangle(tile, p1, p2, c, max(1, z(2)) if not lab.startswith("t") else max(1, z(1)))
                if k in wrong_keys:
                    cv2.rectangle(tile, (p1[0] - z(3), p1[1] - z(3)), (p2[0] + z(3), p2[1] + z(3)), (0, 230, 255), z(2))
                tag(tile, p1[0], max(p1[1], z(14)), text(lab), c)
            cv2.putText(tile, cam, (z(6), z(18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * Z, INK, 1, cv2.LINE_AA)
            y0, x0 = z(30) + row_ * CH, z(10) + col_ * CW
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
            cv2.circle(img, p, z(6), c, -1, cv2.LINE_AA)
            tag(img, p[0] + z(7), p[1] + z(4), text(lab), c, 0.35)
        mm, ss = divmod(f // 30, 60)
        cv2.putText(img, f"game clock {mm}:{ss:02d}  (frame {f})", (z(12), z(22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * Z, INK, 1,
                    cv2.LINE_AA)
        if check and f <= check[0]:
            cv2.rectangle(img, (z(10), z(618)), (z(960), z(646)), (0, 120, 160), -1)
            cv2.putText(img, check[1] + "  (wrong boxes outlined yellow on that frame)", (z(18), z(637)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * Z, INK, 1, cv2.LINE_AA)
        if ff is None:
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        else:
            ff.stdin.write(img.tobytes())
        if f % 1800 == 0:
            print(f"frame {f}", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    print(out)


if __name__ == "__main__":
    main()
