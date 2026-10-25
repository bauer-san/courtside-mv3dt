"""Benchmark the text-detector stage of the jersey reader on torso crops (no DeepStream run needed).

Torso crops as DeepStream makes them (apps/court_filter/torso_objects.c: 12-62 % of player boxes >= 150 px, not cut by
the frame) are sampled from appearance.npz, run through the PP-OCRv4 text detector (ONNX) at several input
configurations with DB post-processing as in apps/jersey (thresh 0.3, box 0.5, unclip 1.6, 2x2 dilation, <= 3 boxes),
and each box is read by the recogniser as DeepStream does now (48 x 64, black padding; tools/jersey_rec_bench.py).
Reference: the full RapidOCR pipeline (tools/jersey.py). Label per tracklet: majority of the Python reads.

  python -m tools.jersey_det_bench <run> [--n 500] [--max-frame 12600]
  python -m tools.jersey_det_bench <run> --emit "letterbox 512x640" --out jersey_emu512.npz
      emulated DeepStream reads (that detector input + the current recogniser) on the same crops as
      tools.jersey extract (big crops, every 2nd appearance sample), for scoring with the tracker
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import load_appearance, run_dataset
from tools.jersey import load as load_reads, make_reader
from tools.jersey_rec_bench import REC, decode, prep

DET = REC.parent / "ch_PP-OCRv4_det_infer.onnx"


def letterbox(img: np.ndarray, W: int, H: int):
    """nvinfer maintain-aspect-ratio + symmetric-padding: centred, black padding. Returns input, scale, offsets."""
    s = min(W / img.shape[1], H / img.shape[0])
    rw, rh = max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))
    out = np.zeros((H, W, 3), np.uint8)
    ox, oy = (W - rw) // 2, (H - rh) // 2
    out[oy:oy + rh, ox:ox + rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return out, s, ox, oy


def scale_min_side(img: np.ndarray, side: int):
    """RapidOCR: scale so the short side is `side`, sizes rounded to multiples of 32."""
    h, w = img.shape[:2]
    r = side / min(h, w) if min(h, w) < side else 1.0
    H, W = max(32, int(round(h * r / 32) * 32)), max(32, int(round(w * r / 32) * 32))
    return cv2.resize(img, (W, H)), W / w, H / h


def db_boxes(prob: np.ndarray, max_boxes: int = 3) -> list:
    """(x, y, w, h, score) in input pixels: threshold, dilate, components, mean score, unclip."""
    bin_ = cv2.dilate((prob >= 0.3).astype(np.uint8), np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 10:
            continue
        sc = float(prob[lab == i].mean())
        if sc < 0.5:
            continue
        d = area * 1.6 / (2 * (w + h))          # unclip: polygon offset by area * ratio / perimeter
        out.append((x - d, y - d, w + 2 * d, h + 2 * d, sc))
    return sorted(out, key=lambda b: -b[4])[:max_boxes]


CONFIGS = {
    "DS now: letterbox 384x480": ("letterbox", (384, 480), 1.0),
    "letterbox 512x640": ("letterbox", (512, 640), 1.0),
    "letterbox 384x480, centre 70 % width": ("letterbox", (384, 480), 0.7),
    "RapidOCR scale (short side 736)": ("minside", 736, 1.0),
    "short side 480": ("minside", 480, 1.0),
}


def read_torso(img, bbox, cfg, det, rec):
    """Emulated DeepStream read of one player box: (number, confidence)."""
    di, ri = det.get_inputs()[0].name, rec.get_inputs()[0].name
    kind, size, keep_w = cfg
    l, t, r, b = (int(v) for v in bbox)
    H = b - t
    torso = img[t + int(0.12 * H): t + int(0.62 * H), max(0, l):r]
    if torso.size == 0:
        return "", 0.0
    x_off = 0
    src = torso
    if keep_w < 1.0:
        w = torso.shape[1]
        x_off = int(w * (1 - keep_w) / 2)
        src = torso[:, x_off:w - x_off]
    if kind == "letterbox":
        x, s, ox, oy = letterbox(src, *size)
        sx = sy = s
    else:
        x, sx, sy = scale_min_side(src, size)
        ox = oy = 0
    inp = ((x.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    prob = det.run(None, {di: inp})[0][0, 0]
    best = ("", 0.0)
    for bx, by, bw, bh, _ in db_boxes(prob):
        fx = l + x_off + (bx - ox) / sx
        fy = t + int(0.12 * H) + (by - oy) / sy
        xin = prep(img, (fx, fy, bw / sx, bh / sy), W=64, pad="black", expand=0.0, fill=False)
        if xin is None:
            continue
        nn, cf = decode(rec.run(None, {ri: xin[None]})[0][0])
        if nn and cf > best[1]:
            best = (nn, cf)
    return best


def emit(run: Path, cfg_name: str, out_name: str, max_frame: int | None, every: int = 2) -> Path:
    from tools.jersey import big_crops
    a = load_appearance(run)
    sel = big_crops(a)
    if max_frame is not None:
        sel &= a["frame"] <= max_frame
    idx = np.where(sel)[0][::every]
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    det, rec = ort.InferenceSession(str(DET), providers=prov), ort.InferenceSession(str(REC), providers=prov)
    want = collections.defaultdict(list)
    for i in idx:
        want[(str(a["cam"][i]), int(a["frame"][i]))].append(i)
    num = np.full(len(a["frame"]), "", dtype=object)
    score = np.zeros(len(a["frame"]), np.float32)
    done = np.zeros(len(a["frame"]), bool)
    ds = run_dataset(run)
    for cam_id, pos, _ in CAMERAS:
        frames = sorted(f for (c, f) in want if c == pos)
        if not frames:
            continue
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        targets, cur = set(frames), 0
        while cur <= frames[-1]:
            fr = cur
            ok, img = (cap.read() if fr in targets else (cap.grab(), None))
            if not ok:
                break
            cur += 1
            if img is None:
                continue
            for i in want[(pos, fr)]:
                num[i], score[i] = read_torso(img, a["bbox"][i], CONFIGS[cfg_name], det, rec)
                done[i] = True
        cap.release()
        print(f"{pos}: {int(done[a['cam'] == pos].sum())} crops, {int(((num != '') & (a['cam'] == pos)).sum())} reads",
              flush=True)
    out = run / out_name
    np.savez_compressed(out, frame=a["frame"][done], cam=a["cam"][done], gid=a["gid"][done],
                        number=num[done].astype(str), score=score[done])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=500, help="torso crops per camera")
    ap.add_argument("--max-frame", type=int, default=12600)
    ap.add_argument("--emit", choices=list(CONFIGS), help="write emulated reads for the whole run instead")
    ap.add_argument("--out", default="jersey_emu.npz")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    if args.emit:
        print(emit(run, args.emit, args.out, args.max_frame))
        return
    py = load_reads(run, "jersey_py.npz")
    votes = collections.defaultdict(collections.Counter)
    for f, c, g, n in zip(py["frame"], py["cam"], py["gid"], py["number"]):
        if n:
            votes[(str(c), str(g))][str(n)] += 1
    label = {k: v.most_common(1)[0][0] for k, v in votes.items()
             if sum(v.values()) >= 3 and v.most_common(1)[0][1] / sum(v.values()) >= 0.7}
    a = load_appearance(run)
    h = a["bbox"][:, 3] - a["bbox"][:, 1]
    big = (~a["cut"]) & (h >= 150) & (a["frame"] <= args.max_frame)
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    det, rec = ort.InferenceSession(str(DET), providers=prov), ort.InferenceSession(str(REC), providers=prov)
    di, ri = det.get_inputs()[0].name, rec.get_inputs()[0].name
    rapid = make_reader()
    rng = np.random.default_rng(1)
    ds = run_dataset(run)
    res = collections.defaultdict(collections.Counter)
    ms = collections.defaultdict(list)
    for cam_id, pos, _ in CAMERAS:
        idx = np.where(big & (a["cam"] == pos))[0]
        idx = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        idx = idx[np.argsort(a["frame"][idx], kind="stable")]
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        cur = -1
        for i in idx:
            fr = int(a["frame"][i])
            while cur < fr:
                cap.grab()
                cur += 1
            ok, img = cap.retrieve()
            if not ok:
                continue
            l, t, r, b = (int(v) for v in a["bbox"][i])
            H = b - t
            torso = img[t + int(0.12 * H): t + int(0.62 * H), max(0, l):r]
            if torso.size == 0:
                continue
            truth = label.get((pos, str(a["gid"][i])))
            n, _ = rapid(img[max(0, t):b, max(0, l):r])
            c = res[(pos, "RapidOCR full pipeline (reference)")]
            c["n"] += 1
            if n:
                c["read"] += 1
                if truth:
                    c["ok" if n == truth else "wrong"] += 1
            for name, (kind, size, keep_w) in CONFIGS.items():
                src = torso
                x_off = 0
                if keep_w < 1.0:
                    w = torso.shape[1]
                    x_off = int(w * (1 - keep_w) / 2)
                    src = torso[:, x_off:w - x_off]
                if kind == "letterbox":
                    x, s, ox, oy = letterbox(src, *size)
                    sx = sy = s
                else:
                    x, sx, sy = scale_min_side(src, size)
                    ox = oy = 0
                inp = ((x.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
                t0 = time.perf_counter()
                prob = det.run(None, {di: inp})[0][0, 0]
                ms[name].append(inp.shape[2] * inp.shape[3])
                best = ("", 0.0)
                for bx, by, bw, bh, _ in db_boxes(prob):
                    # back to frame coordinates
                    fx = l + x_off + (bx - ox) / sx
                    fy = t + int(0.12 * H) + (by - oy) / sy
                    xin = prep(img, (fx, fy, bw / sx, bh / sy), W=64, pad="black", expand=0.0, fill=False)
                    if xin is None:
                        continue
                    nn, cf = decode(rec.run(None, {ri: xin[None]})[0][0])
                    if nn and cf > best[1]:
                        best = (nn, cf)
                c = res[(pos, name)]
                c["n"] += 1
                if best[0]:
                    c["read"] += 1
                    if truth:
                        c["ok" if best[0] == truth else "wrong"] += 1
        cap.release()
        print(f"{pos}: {len(idx)} torso crops", flush=True)
    cams = [p for _, p, _ in CAMERAS]
    names = ["RapidOCR full pipeline (reference)"] + list(CONFIGS)
    print(f"\n{'detector input':40s} " + "  ".join(f"{c:>17s}" for c in cams + ["all"]) + "   pixels vs DS")
    base = np.mean(ms["DS now: letterbox 384x480"])
    for name in names:
        tot, cells = collections.Counter(), []
        for c in cams:
            r = res[(c, name)]
            tot.update(r)
            cells.append(f"{r['read'] / max(r['n'], 1):4.0%} rd {r['wrong'] / max(r['ok'] + r['wrong'], 1):3.0%} wr")
        cells.append(f"{tot['read'] / max(tot['n'], 1):4.0%} rd {tot['wrong'] / max(tot['ok'] + tot['wrong'], 1):3.0%} wr")
        px = f"{np.mean(ms[name]) / base:5.1f}x" if name in ms else ""
        print(f"{name:40s} " + "  ".join(f"{x:>17s}" for x in cells) + f"   {px}")


if __name__ == "__main__":
    main()
