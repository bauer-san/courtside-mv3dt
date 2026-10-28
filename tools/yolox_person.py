"""Offline check of YOLOX as the person detector in place of PeopleNet (GitHub issue #10, step 1).

YOLOX-m / YOLOX-l (models/yolox, Apache-2.0, COCO; fixed 640 x 640 input, the 1920 x 1080 frame letterboxed, as the
official demo) on frames of a default DeepStream run's time-aligned clips, against that run's PeopleNet boxes (court
filtered, as recorded on Kafka). YOLOX boxes get the same court filter (box-bottom floor point within the court plus
0.75 m, or the box cut by the bottom image edge).

  labels       labelled detections (gt/*, carried over to the run): share of each category (player, referee, not a
               player, unclear) that YOLOX also boxes (IoU >= 0.5), per score threshold
  boxes        court-filtered boxes per frame, and YOLOX boxes overlapping no PeopleNet box (IoU < 0.3)
  placement    box bottom and top of YOLOX minus PeopleNet, matched player boxes, per camera
  consistency  the same person in two cameras at the same frame (court-tracker track of the run, one frame per second
               of the game): distance between their box-bottom floor points, PeopleNet boxes vs the matched YOLOX
               boxes on the same pairs (boxes cut by the image edge left out)

  python -m tools.yolox_person js_r1 [--models yolox_m yolox_l] [--md out.md]
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import load_boxes, run_dataset
from tools.evaluate import inside
from tools.gt_eval import game_end
from tools.stitch import floor_point, load_cameras
from tools.variant_eval import iou_matrix

THRESHOLDS = (0.1, 0.25, 0.4)
SIZE = 640
NMS_IOU = 0.45
MARGIN = 0.75
FRAME_H = 1080


def session(model: str):
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    return ort.InferenceSession(str(ROOT / "models" / "yolox" / f"{model}.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


_GRID = None


def decode(out: np.ndarray) -> np.ndarray:
    """YOLOX raw output (8400, 85) for a 640 x 640 input -> boxes as cx, cy, w, h in input pixels."""
    global _GRID
    if _GRID is None:
        grids, strides = [], []
        for s in (8, 16, 32):
            n = SIZE // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        _GRID = (np.concatenate(grids), np.concatenate(strides))
    g, st = _GRID
    out = out.copy()
    out[:, :2] = (out[:, :2] + g) * st
    out[:, 2:4] = np.exp(out[:, 2:4]) * st
    return out


def detect(sess, img: np.ndarray, min_score: float = min(THRESHOLDS)) -> np.ndarray:
    """Person boxes [l, t, r, b, score] in image pixels (letterboxed to 640 x 640, NMS IoU 0.45)."""
    import cv2
    r = min(SIZE / img.shape[0], SIZE / img.shape[1])
    small = cv2.resize(img, (round(img.shape[1] * r), round(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
    pad = np.full((SIZE, SIZE, 3), 114, np.uint8)
    pad[:small.shape[0], :small.shape[1]] = small
    out = decode(sess.run(None, {"images": pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0])
    sc = out[:, 4] * out[:, 5]                      # objectness x "person" (COCO class 0)
    i = np.where(sc >= min_score)[0]
    if not len(i):
        return np.zeros((0, 5))
    cx, cy, w, h = out[i, :4].T
    b = np.c_[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] / r
    keep = cv2.dnn.NMSBoxes([[float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])] for x in b],
                            sc[i].tolist(), min_score, NMS_IOU)
    keep = np.array(keep).reshape(-1)
    return np.c_[b[keep], sc[i][keep]]


def court(boxes: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Mask of boxes kept by the court filter (box-bottom floor point within the court + 0.75 m, or cut at the bottom)."""
    if not len(boxes):
        return np.zeros(0, bool)
    xy, ok = floor_point(P, (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3])
    return (ok & inside(xy, MARGIN)) | (boxes[:, 3] >= FRAME_H - 5)


def match(A: np.ndarray, B: np.ndarray, thr: float = 0.5) -> dict:
    """Greedy one-to-one matches {index in A: index in B} with IoU >= thr."""
    out = {}
    if not len(A) or not len(B):
        return out
    M = iou_matrix(A[:, :4], B[:, :4])
    while True:
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] < thr:
            return out
        out[i] = j
        M[i, :] = -1
        M[:, j] = -1


def frames(run: Path) -> tuple[dict, dict, dict]:
    """Labelled detections, sampled pairs and the frames to decode per camera."""
    from tools.nonplayer_eval import categories
    from tools.world_tracker import track_default
    lab = categories(run)                                        # (frame, cam, gid) -> (category, label)
    end = game_end(run)
    sample = set(range(300, end + 1, 30))
    pairs = []                                                   # (frame, (cam, gid), (cam, gid)) same court track
    for t in track_default(run):
        by = collections.defaultdict(dict)
        for f, c, g in t.dets:
            if f in sample:
                by[f][c] = str(g)
        for f, cg in by.items():
            cs = sorted(cg)
            pairs += [(f, (a, cg[a]), (b, cg[b])) for i, a in enumerate(cs) for b in cs[i + 1:]]
    need = collections.defaultdict(set)
    for f, c, _ in lab:
        need[c].add(f)
    for f, a, b in pairs:
        need[a[0]].add(f)
        need[b[0]].add(f)
    return lab, pairs, need


def run_models(run: Path, models: list[str], need: dict) -> dict:
    """(model, cam, frame) -> YOLOX boxes, decoding each camera's clip once."""
    import cv2
    ds = run_dataset(run)
    sess = {m: session(m) for m in models}
    out = {}
    for cam_id, pos, _ in CAMERAS:
        want = need.get(pos, set())
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        f, last = 0, max(want, default=-1)
        while f <= last:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                for m, s in sess.items():
                    out[(m, pos, f)] = detect(s, img)
            f += 1
        cap.release()
    return out


def evaluate(run_name: str, models: list[str]) -> str:
    run = ROOT / "experiments" / run_name
    cams = load_cameras(run / "camInfo_used")
    pn = {(c, f): np.array([b[1:] for b in v], float) for (c, f), v in load_boxes(run, 1).items()}
    gids = {(c, f): [str(b[0]) for b in v] for (c, f), v in load_boxes(run, 1).items()}
    lab, pairs, need = frames(run)
    Y = run_models(run, models, need)
    lines = [f"### YOLOX vs PeopleNet boxes on {run_name} (time-aligned clips; YOLOX 640 x 640 letterboxed)", ""]

    # labelled detections found
    lines += ["| detector, score threshold | " + " | ".join(
        ["player", "referee", "not a player", "unclear"]) + " | court-filtered boxes per frame | boxes on no PeopleNet box, per frame |",
        "|---|---|---|---|---|---|---|"]
    keys = sorted({(c, f) for f, c, _ in lab})
    n_pn = sum(len(pn.get(k, ())) for k in keys) / max(len(keys), 1)
    lines.append(f"| PeopleNet (as run) | 100 % | 100 % | 100 % | 100 % | {n_pn:.2f} | - |")
    for m in models:
        for thr in THRESHOLDS:
            hit, tot, nb, extra = collections.Counter(), collections.Counter(), 0, 0
            for c, f in keys:
                y = Y[(m, c, f)]
                y = y[(y[:, 4] >= thr) & court(y, cams[c])] if len(y) else y
                nb += len(y)
                p = pn.get((c, f), np.zeros((0, 4)))
                if len(y) and len(p):
                    extra += int((iou_matrix(y[:, :4], p).max(1) < 0.3).sum())
                else:
                    extra += len(y)
                mt = match(p, y) if len(y) else {}
                for i, g in enumerate(gids.get((c, f), [])):
                    cat = lab.get((f, c, g))
                    if cat:
                        tot[cat[0]] += 1
                        hit[cat[0]] += i in mt
            cells = [f"{hit[k] / tot[k]:.1%} ({tot[k]})" if tot[k] else "-"
                     for k in ("player", "referee", "not a player", "unclear")]
            lines.append(f"| {m}, {thr} | " + " | ".join(cells) + f" | {nb / len(keys):.2f} | {extra / len(keys):.2f} |")

    # placement and cross-camera consistency (threshold 0.25)
    for m in models:
        dtop, dbot = collections.defaultdict(list), collections.defaultdict(list)
        d_pn, d_y = collections.defaultdict(list), collections.defaultdict(list)
        cache = {}

        def ybox(c, f, g):
            k = (c, f)
            if k not in cache:
                y = Y.get((m, c, f), np.zeros((0, 5)))
                y = y[y[:, 4] >= 0.25] if len(y) else y
                p = pn.get(k, np.zeros((0, 4)))
                cache[k] = (match(p, y), y)
            mt, y = cache[k]
            i = gids[k].index(g)
            return pn[k][i], (y[mt[i], :4] if i in mt else None)

        for f, a, b in pairs:
            pa, ya = ybox(a[0], f, a[1])
            pb, yb = ybox(b[0], f, b[1])
            for c, p, y in ((a[0], pa, ya), (b[0], pb, yb)):
                if y is not None:
                    dtop[c].append(y[1] - p[1])
                    dbot[c].append(y[3] - p[3])
            if ya is None or yb is None or max(pa[3], pb[3], ya[3], yb[3]) >= FRAME_H - 5:
                continue
            pt = lambda c, x: floor_point(cams[c], np.array([(x[0] + x[2]) / 2]), np.array([x[3]]))[0][0]
            key = "-".join(sorted((a[0], b[0])))
            d_pn[key].append(np.linalg.norm(pt(a[0], pa) - pt(b[0], pb)))
            d_y[key].append(np.linalg.norm(pt(a[0], ya) - pt(b[0], yb)))
        lines += ["", f"**{m}, threshold 0.25: box placement (YOLOX minus PeopleNet, px, median) and cross-camera "
                      "distance of box-bottom floor points (median, m)**", "",
                  "| camera | top | bottom | matched |", "|---|---|---|---|"]
        for c in ("NL", "FL", "NR", "FR"):
            if dbot[c]:
                lines.append(f"| {c} | {np.median(dtop[c]):+.1f} | {np.median(dbot[c]):+.1f} | {len(dbot[c])} |")
        lines += ["", "| camera pair | pairs | PeopleNet | YOLOX |", "|---|---|---|---|"]
        for k in sorted(d_pn):
            lines.append(f"| {k} | {len(d_pn[k])} | {np.median(d_pn[k]):.2f} | {np.median(d_y[k]):.2f} |")
        allp, ally = sum(d_pn.values(), []), sum(d_y.values(), [])
        lines.append(f"| all | {len(allp)} | {np.median(allp):.2f} | {np.median(ally):.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--models", nargs="+", default=["yolox_m", "yolox_l"])
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    md = evaluate(a.run, a.models)
    print(md)
    if a.md:
        a.md.write_text(md)


if __name__ == "__main__":
    main()
