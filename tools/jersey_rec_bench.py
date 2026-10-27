"""Benchmark recogniser preprocessing on DeepStream's own text boxes (no DeepStream run needed).

Samples text boxes that DeepStream's text detector found (jersey_reads.csv of a JERSEY=1 run), crops them from the
dataset videos and runs the PP-OCRv4 recogniser (ONNX, onnxruntime) with several preprocessing variants, decoding as
apps/jersey/nvdsinfer_jersey_parsers.cpp does (greedy CTC, 1-2 digits, mean max probability >= MIN_CONF).
Reference label per tracklet (camera + tracker id): the majority of the independent Python reads (jersey_py.npz).

  python -m tools.jersey_rec_bench <run> [--n 600] [--max-frame GAME_END]
"""

import argparse
import collections
import csv
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, GAME_END, ROOT
from tools.appearance import run_dataset
from tools.jersey import load as load_reads

REC = ROOT / "external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/OCR" \
    / "ch_PP-OCRv4_rec_infer.onnx"
DIGITS = {26: "0", 93: "1", 25: "2", 94: "3", 632: "4", 631: "5", 933: "6", 29: "7", 27: "8", 1109: "9"}
MIN_CONF = 0.6


def decode(prob: np.ndarray) -> tuple[str, float]:
    idx = prob.argmax(1)
    text, confs, prev = "", [], -1
    for t, i in enumerate(idx):
        if i != 0 and i != prev:
            c = DIGITS.get(int(i))
            if c is None:
                return "", 0.0
            text += c
            confs.append(prob[t, i])
        prev = i
    if not 1 <= len(text) <= 2:
        return "", 0.0
    conf = float(np.mean(confs))
    return (text, conf) if conf >= MIN_CONF else ("", conf)


def prep(img: np.ndarray, box: tuple, W: int, pad: str, expand: float, fill: bool) -> np.ndarray:
    """Crop a text box and lay it into a 48 x W input (BGR, (x - 127.5) / 127.5).
    expand: extra margin around the box (fraction of its height); fill: widen/heighten the crop with image context
    to the input aspect instead of padding."""
    l, t, w, h = box
    cx, cy = l + w / 2, t + h / 2
    w, h = w + 2 * expand * h, h * (1 + 2 * expand)
    if fill:
        if w / h < W / 48:
            w = h * W / 48
        else:
            h = w * 48 / W
    H_img, W_img = img.shape[:2]
    x0, x1 = int(max(0, cx - w / 2)), int(min(W_img, cx + w / 2))
    y0, y1 = int(max(0, cy - h / 2)), int(min(H_img, cy + h / 2))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    s = min(W / crop.shape[1], 48 / crop.shape[0])
    rw, rh = max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s)))
    r = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = np.full((48, W, 3), 0.0 if pad == "black" else 127.5, np.float32)
    out[:rh, :rw] = r                          # nvinfer symmetric-padding=0: image top-left, pad right/bottom
    return ((out - 127.5) / 127.5).transpose(2, 0, 1)


VARIANTS = {
    "DS now: 48x64, black pad": dict(W=64, pad="black", expand=0.0, fill=False),
    "48x96, black pad": dict(W=96, pad="black", expand=0.0, fill=False),
    "48x160, black pad": dict(W=160, pad="black", expand=0.0, fill=False),
    "48x320, grey pad (RapidOCR-like)": dict(W=320, pad="grey", expand=0.0, fill=False),
    "48x64, filled with shirt": dict(W=64, pad="black", expand=0.0, fill=True),
    "48x96, filled with shirt": dict(W=96, pad="black", expand=0.0, fill=True),
    "48x64, +15% margin, filled": dict(W=64, pad="black", expand=0.15, fill=True),
    "48x96, +15% margin, filled": dict(W=96, pad="black", expand=0.15, fill=True),
    "48x96, +30% margin, filled": dict(W=96, pad="black", expand=0.30, fill=True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=600, help="text boxes per camera")
    ap.add_argument("--max-frame", type=int, default=GAME_END)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    cams = [pos for _, pos, _ in CAMERAS]
    py = load_reads(run, "jersey_py.npz")
    votes = collections.defaultdict(collections.Counter)
    for f, c, g, n in zip(py["frame"], py["cam"], py["gid"], py["number"]):
        if n:
            votes[(str(c), str(g))][str(n)] += 1
    label = {k: v.most_common(1)[0][0] for k, v in votes.items()
             if sum(v.values()) >= 3 and v.most_common(1)[0][1] / sum(v.values()) >= 0.7}
    rows = collections.defaultdict(list)
    with (run / "jersey_reads.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["frame"]) <= args.max_frame and float(r["tw"]) > 1 and float(r["th"]) > 1:
                rows[cams[int(r["source"])]].append(r)
    rng = np.random.default_rng(0)
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(str(REC), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    ds = run_dataset(run)
    res = collections.defaultdict(lambda: collections.Counter())
    for cam_id, pos, _ in CAMERAS:
        pick = [rows[pos][i] for i in sorted(rng.choice(len(rows[pos]), min(args.n, len(rows[pos])), replace=False))]
        pick.sort(key=lambda r: int(r["frame"]))
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        cur, img = -1, None
        for r in pick:
            fr = int(r["frame"])
            while cur < fr:
                ok = cap.grab()
                cur += 1
            ok, img = cap.retrieve()
            if not ok:
                continue
            box = (float(r["tl"]), float(r["tt"]), float(r["tw"]), float(r["th"]))
            truth = label.get((pos, r["track_id"]))
            res[(pos, "DeepStream actual")]["n"] += 1
            if r["number"]:
                res[(pos, "DeepStream actual")]["read"] += 1
                if truth:
                    res[(pos, "DeepStream actual")]["ok" if r["number"] == truth else "wrong"] += 1
            for name, v in VARIANTS.items():
                x = prep(img, box, **v)
                if x is None:
                    continue
                n, _ = decode(sess.run(None, {inp: x[None]})[0][0])
                c = res[(pos, name)]
                c["n"] += 1
                if n:
                    c["read"] += 1
                    if truth:
                        c["ok" if n == truth else "wrong"] += 1
        cap.release()
        print(f"{pos}: {len(pick)} text boxes", flush=True)
    names = ["DeepStream actual"] + list(VARIANTS)
    print(f"\n{'variant':34s} " + "  ".join(f"{c:>18s}" for c in cams + ["all"]))
    for name in names:
        cells = []
        tot = collections.Counter()
        for c in cams:
            r = res[(c, name)]
            tot.update(r)
            cells.append(f"{r['read'] / max(r['n'], 1):4.0%} read, {r['wrong'] / max(r['ok'] + r['wrong'], 1):3.0%} wr")
        cells.append(f"{tot['read'] / max(tot['n'], 1):4.0%} read, {tot['wrong'] / max(tot['ok'] + tot['wrong'], 1):3.0%} wr")
        print(f"{name:34s} " + "  ".join(f"{x:>18s}" for x in cells))


if __name__ == "__main__":
    main()
