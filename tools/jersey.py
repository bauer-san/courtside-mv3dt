"""Jersey-number reads with RapidOCR (PaddleOCR PP-OCRv4 text detector + recogniser, ONNX, Apache-2.0).

extract   for big crops in appearance.npz (not cut, box >= MIN_H px, all cameras; the steep backboard
          cameras NL/NR read as well as the end-wall ones on such crops) up to --max-frame, taking every `--every`-th sample,
          read the torso (12-62 % of the box height, rescaled to 160 px) and keep a result only if it is 1-2
          digits with recogniser score >= MIN_SCORE
          -> experiments/<run>/jersey.npz  (frame, cam, gid, number ('' = no read), score)
Feasibility on 59 hand-labelled crops: 20/26 legible numbers read, 0 wrong, 1 doubtful read on 33 illegible crops.

  python -m tools.jersey extract <run> [--max-frame GAME_END] [--every 2] [--out jersey.npz]
  python -m tools.jersey from-ds <run>      # convert DeepStream's jersey_reads.csv (JERSEY=1 runs)

from-ds   the deployable path: reads produced inside DeepStream (torso objects -> text-detector SGIE ->
          digit-recogniser SGIE, apps/jersey + apps/court_filter). One entry per (frame, camera, tracker id)
          that had a text box; the most confident digit read wins. Same jersey.npz format as `extract`.
"""

import argparse
import collections
import re
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, GAME_END, ROOT
from tools.appearance import load_appearance, run_dataset

MIN_H = 150
CAMS = ("NL", "FL", "NR", "FR")   # all four: NL read rate 26 % vs NR 21 % on big crops (tested 2026-10-22)
MIN_SCORE = 0.6


def big_crops(a: dict) -> np.ndarray:
    h = a["bbox"][:, 3] - a["bbox"][:, 1]
    return (~a["cut"]) & (h >= MIN_H) & np.isin(a["cam"], list(CAMS))


def torso(crop: np.ndarray) -> np.ndarray:
    H = crop.shape[0]
    t = crop[int(0.12 * H):int(0.62 * H), :]
    s = 160.0 / max(t.shape[0], 1)
    return cv2.resize(t, (max(1, int(t.shape[1] * s)), 160), interpolation=cv2.INTER_CUBIC)


def make_reader():
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)

    def read(crop: np.ndarray) -> tuple[str, float]:
        res, _ = ocr(torso(crop), use_cls=False)
        best = ("", 0.0)
        for _, text, score in res or []:
            t = text.strip().replace(" ", "")
            if re.fullmatch(r"[0-9]{1,2}", t) and score >= MIN_SCORE and score > best[1]:
                best = (t, float(score))
        return best

    return read


def extract(run: Path, max_frame: int | None = None, every: int = 2, out_name: str = "jersey.npz") -> Path:
    a = load_appearance(run)
    sel = big_crops(a)
    if max_frame is not None:
        sel &= a["frame"] <= max_frame
    idx = np.where(sel)[0][::every]
    want = collections.defaultdict(list)
    for i in idx:
        want[(str(a["cam"][i]), int(a["frame"][i]))].append(i)
    read = make_reader()
    ds = run_dataset(run)
    num = np.full(len(a["frame"]), "", dtype=object)
    score = np.zeros(len(a["frame"]), np.float32)
    done = np.zeros(len(a["frame"]), bool)
    for cam_id, pos, _ in CAMERAS:
        frames = sorted(f for (c, f) in want if c == pos)
        if not frames:
            continue
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        targets, cur = set(frames), 0
        while cur <= frames[-1]:
            fr = cur                # kafka frame id = video frame index
            ok, img = (cap.read() if fr in targets else (cap.grab(), None))
            if not ok:
                break
            cur += 1
            if img is None:
                continue
            for i in want[(pos, fr)]:
                l, t, r, b = (int(v) for v in a["bbox"][i])
                n, s = read(img[max(0, t):b, max(0, l):r])
                num[i], score[i], done[i] = n, s, True
        cap.release()
        print(f"{pos}: {int(done[a['cam'] == pos].sum())} crops, {int(((num != '') & (a['cam'] == pos)).sum())} reads",
              flush=True)
    out = run / out_name
    np.savez_compressed(out, frame=a["frame"][done], cam=a["cam"][done], gid=a["gid"][done],
                        number=num[done].astype(str), score=score[done])
    return out


def from_ds(run: Path) -> Path:
    import csv
    cams = [pos for _, pos, _ in CAMERAS]          # DeepStream source id -> camera (stream order)
    best = {}
    with (run / "jersey_reads.csv").open() as f:
        for r in csv.DictReader(f):
            k = (int(r["frame"]), cams[int(r["source"])], r["track_id"])
            n, sc = r["number"], float(r["conf"]) if r["number"] else 0.0
            if k not in best or sc > best[k][1]:
                best[k] = (n, sc)
    keys = sorted(best)
    out = run / "jersey.npz"
    np.savez_compressed(out, frame=np.array([k[0] for k in keys]), cam=np.array([k[1] for k in keys]),
                        gid=np.array([k[2] for k in keys]), number=np.array([best[k][0] for k in keys]),
                        score=np.array([best[k][1] for k in keys], np.float32))
    return out


def load(run: Path, name: str = "jersey.npz") -> dict:
    with np.load(run / name) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "from-ds"])
    ap.add_argument("run")
    ap.add_argument("--max-frame", type=int, default=None)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", default="jersey.npz", help="extract: output file name in the run directory")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    print(extract(run, args.max_frame, args.every, args.out) if args.cmd == "extract" else from_ds(run))


if __name__ == "__main__":
    main()
