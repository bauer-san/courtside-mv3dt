"""Where two detector variants put the same person's box, per camera (GitHub issue #1: INT8 vs FP16).

Runs are paired (a[i] with b[i]). On the game frames, court-filtered boxes of the same frame and camera are matched
one to one (greedy, IoU >= 0.5); for each match, b minus a:
  top, bottom    box edges in pixels (+ = lower in the image)
  height         relative change of the box height
  foot (m)       MV3DT foot point, along the camera's viewing direction (+ = farther from the camera) and across it
Plus boxes per frame and camera on each side, and the share matched. Two runs of the same variant give the noise
floor (runs are not repeatable).

  python -m tools.box_compare --a js_r1 js_r2 js_r3 --b fp16t10_r1 fp16t10_r2 fp16t10_r3 [--md out.md]
  python -m tools.box_compare --a js_r1 --b js_r2          # noise floor
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.appearance import load_boxes
from tools.evaluate import inside
from tools.gt_eval import game_end
from tools.stitch import Params as StitchParams, ground_filter, load_cameras, load_detections
from tools.variant_eval import iou_matrix

CAMS = ("NL", "FL", "NR", "FR")


def boxes(run: Path) -> dict:
    """(frame, cam) -> (N x 4 boxes [l, t, r, b], N x 2 MV3DT foot points) for court-filtered game detections."""
    last = game_end(run)
    d = load_detections(run / "kafka.jsonl.gz")
    keep = ground_filter(d, load_cameras(run / "camInfo_used"), inside, StitchParams(margin=0.3))
    foot = {(int(f), str(c), str(g)): (x, y) for f, c, g, x, y, k in
            zip(d["frame"], d["cam"], d["gid"], d["x"], d["y"], keep) if k and f <= last}
    out = collections.defaultdict(lambda: ([], []))
    for (cam, fr), v in load_boxes(run, 1).items():
        for g, l, t, r, b in v:
            p = foot.get((fr, cam, str(g)))
            if p is not None:
                out[(fr, cam)][0].append((l, t, r, b))
                out[(fr, cam)][1].append(p)
    return {k: (np.array(B, float), np.array(F, float)) for k, (B, F) in out.items()}


def pairs(A: dict, B: dict, thr: float = 0.5):
    """Yield (cam, box_a, box_b, foot_a, foot_b) for greedy one-to-one matches with IoU >= thr."""
    for k in A.keys() & B.keys():
        (ba, fa), (bb, fb) = A[k], B[k]
        M = iou_matrix(ba, bb)
        while M.size:
            i, j = np.unravel_index(np.argmax(M), M.shape)
            if M[i, j] < thr:
                break
            yield k[1], ba[i], bb[j], fa[i], fb[j]
            M[i, :] = -1
            M[:, j] = -1


def compare(a: str, b: str) -> dict:
    """cam -> dict of per-match difference arrays and box counts."""
    A, B = boxes(ROOT / "experiments" / a), boxes(ROOT / "experiments" / b)
    cams = load_cameras(ROOT / "experiments" / a / "camInfo_used")
    centre = {c: -np.linalg.solve(P[:, :3], P[:, 3])[:2] for c, P in cams.items()}   # camera position on the floor
    res = {c: collections.defaultdict(list) for c in CAMS}
    for cam, x, y, fx, fy in pairs(A, B):
        r = res[cam]
        r["top"].append(y[1] - x[1])
        r["bottom"].append(y[3] - x[3])
        r["height"].append((y[3] - y[1]) / max(x[3] - x[1], 1) - 1)
        u = fx - centre[cam]
        u = u / (np.linalg.norm(u) + 1e-9)
        dv = fy - fx
        r["along"].append(float(dv @ u))
        r["across"].append(float(dv[0] * -u[1] + dv[1] * u[0]))
    keys = A.keys() & B.keys()
    for c in CAMS:
        ks = [k for k in keys if k[1] == c]
        res[c]["n_a"] = sum(len(A[k][0]) for k in ks)
        res[c]["n_b"] = sum(len(B[k][0]) for k in ks)
        res[c]["frames"] = len(ks)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    assert len(a.a) == len(a.b), "runs are compared in pairs"
    tot = {c: collections.defaultdict(list) for c in CAMS}
    for x, y in zip(a.a, a.b):
        for c, r in compare(x, y).items():
            for k, v in r.items():
                tot[c][k] += v if isinstance(v, list) else [v]
    med = lambda v: f"{np.median(v):+.1f}" if v else "-"
    lines = [f"### Box placement, {' '.join(a.b)} minus {' '.join(a.a)} (matched boxes, IoU >= 0.5, game frames)", "",
             "| camera | boxes per frame (a / b) | matched | top px (median, IQR) | bottom px (median, IQR) | "
             "height (median) | foot along view, m (median, IQR) | foot across, m (median) |",
             "|---|---|---|---|---|---|---|---|"]
    for c in CAMS:
        r = tot[c]
        if not r["top"]:
            continue
        fr = sum(r["frames"])
        iqr = lambda v, s=1: f"{np.percentile(v, 25) * s:+.2f}..{np.percentile(v, 75) * s:+.2f}"
        lines.append(f"| {c} | {sum(r['n_a']) / fr:.2f} / {sum(r['n_b']) / fr:.2f} | "
                     f"{len(r['top']) / max(sum(r['n_a']), 1):.1%} | {med(r['top'])} ({iqr(r['top'])}) | "
                     f"{med(r['bottom'])} ({iqr(r['bottom'])}) | {np.median(r['height']):+.1%} | "
                     f"{np.median(r['along']):+.3f} ({iqr(r['along'])}) | {np.median(r['across']):+.3f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if a.md:
        a.md.write_text(md)


if __name__ == "__main__":
    main()
