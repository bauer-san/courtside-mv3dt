"""Detection-level comparison of pipeline variants (INT8 detector, detector interval, no display) with FP16 references.

For each variant run against a reference run (detector every frame), on the game frames:
  speed            fps per camera, GPU use and memory (tools/evaluate.perf)
  output frames    share of frames that carry any detections (a detector interval with MV3DT's visual tracker off
                   only outputs on detector frames)
  box recall       share of the reference's court-filtered boxes that the variant also has (IoU >= 0.5, same
                   frame and camera), on the frames where the variant outputs; and the reverse ("box precision")
  label recall     share of hand-labelled people (gt/*, carried over by box overlap) the variant detects
Two FP16 reference runs compared with each other give the noise floor (runs are not repeatable).

  python -m tools.variant_eval --ref mount_r1 --runs mount_r2 int8_r1 int1_r1 [--md out.md]
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.appearance import load_boxes
from tools.evaluate import inside, perf
from tools.gt_eval import game_end, load_gt
from tools.stitch import Params as StitchParams, ground_filter, load_cameras, load_detections

def court_boxes(run: Path) -> dict:
    """(frame, cam) -> array of court-filtered boxes [l, t, r, b] (game frames)."""
    last = game_end(run)
    d = load_detections(run / "kafka.jsonl.gz")
    keep = ground_filter(d, load_cameras(run / "camInfo_used"), inside, StitchParams(margin=0.3))
    ok = {(int(f), str(c), str(g)) for f, c, g in zip(d["frame"][keep], d["cam"][keep], d["gid"][keep])
          if f <= last}
    out = collections.defaultdict(list)
    for (cam, fr), v in load_boxes(run, 1).items():
        if fr > last:
            continue
        for g, l, t, r, b in v:
            if (fr, cam, str(g)) in ok:
                out[(fr, cam)].append((l, t, r, b))
    return {k: np.array(v, float) for k, v in out.items()}


def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (area(A)[:, None] + area(B)[None, :] - inter + 1e-9)


def matched(A: np.ndarray, B: np.ndarray, thr: float = 0.5) -> int:
    """Greedy one-to-one matches between box sets A and B with IoU >= thr."""
    if len(A) == 0 or len(B) == 0:
        return 0
    M = iou_matrix(A, B)
    n = 0
    while True:
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] < thr:
            return n
        n += 1
        M[i, :] = -1
        M[:, j] = -1


def compare(ref: dict, var: dict) -> dict:
    frames_var = {k[0] for k in var}
    keys = [k for k in ref if k[0] in frames_var]
    n_ref = sum(len(ref[k]) for k in keys)
    n_var = sum(len(var.get(k, ())) for k in keys)
    m = sum(matched(ref[k], var.get(k, np.zeros((0, 4)))) for k in keys)
    return dict(box_recall=m / max(n_ref, 1), box_precision=m / max(n_var, 1),
                objects_per_frame_cam_ref=n_ref / max(len(keys), 1),
                objects_per_frame_cam_var=n_var / max(len(keys), 1))


def label_recall(ref_run: Path, run: Path) -> float:
    gt = {**load_gt("game_2to3"), **load_gt("game_4to5")}          # on the labelling run's detections
    src = ROOT / "experiments" / "full_pn263_cf_jersey"
    a = load_boxes(src, 60)
    b = load_boxes(run, 60)
    hit = tot = 0
    for (fr, cam, gid) in gt:
        box = next((x[1:] for x in a.get((cam, fr), []) if str(x[0]) == gid), None)
        if box is None:
            continue
        tot += 1
        cand = np.array([x[1:] for x in b.get((cam, fr), [])], float)
        if len(cand) and iou_matrix(np.array([box], float), cand).max() >= 0.5:
            hit += 1
    return hit / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    rp = lambda r: Path(r) if Path(r).is_dir() else ROOT / "experiments" / r
    ref_run = rp(args.ref)
    ref = court_boxes(ref_run)
    n_frames = game_end(ref_run) + 1
    lines = ["| run | config | fps/camera | GPU use | GPU mem | frames with output | box recall vs ref | "
             "box precision vs ref | label recall |", "|---|---|---|---|---|---|---|---|---|"]
    for r in args.runs:
        run = rp(r)
        cfg = (run / "run.txt").read_text().splitlines()[0]
        short = " ".join(x for x in cfg.split() if x.split("=")[0] in ("display", "pgie_interval", "pgie_precision")
                         or x.startswith("extra="))
        p = perf(run)
        var = court_boxes(run)
        frames_out = len({k[0] for k in var}) / n_frames
        c = compare(ref, var)
        lr = label_recall(ref_run, run)
        lines.append(f"| {r} | {short} | {p.get('fps', float('nan')):.1f} | {p.get('gpu_util_med', float('nan')):.0f} % | "
                     f"{p.get('gpu_mem_peak_mib', float('nan')) / 1024:.2f} GB | {frames_out:.0%} | "
                     f"{c['box_recall']:.1%} | {c['box_precision']:.1%} | {lr:.1%} |")
        print(lines[-1], flush=True)
    md = f"### Detection-level comparison with {args.ref}, game frames\n\n" + "\n".join(lines) + "\n"
    print(md)
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
