"""ReID embeddings (TAO ReIdentificationNet, ONNX) for the court tracker, computed offline on the GPU.

extract   for the good crops in appearance.npz (not cut by the frame, >= MIN_BOX_H px, FL/FR/NR) up to
          --max-frame, crop the box from the video, resize to 128 x 256, RGB, (x - mean) * 0.01735 (the
          preprocessing of NVIDIA's tracker ReID config), run the network, L2-normalise
          -> experiments/<run>/reid.npz  (frame, cam, gid, emb float16 [N, 256])
Needs onnxruntime-gpu; its CUDA/cuDNN pip wheels are loaded with onnxruntime.preload_dlls().

  python -m tools.reid extract <run> [--max-frame 12600]
"""

import argparse
import collections
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import good_crops, load_appearance, run_dataset

MODEL = ROOT / "external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/ReID" \
    / "resnet50_market1501_aicity156.onnx"
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
SCALE = 0.01735207
BATCH = 64


def session():
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    return ort.InferenceSession(str(MODEL), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    x = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32)
    return ((x - MEAN) * SCALE).transpose(2, 0, 1)


def extract(run: Path, max_frame: int | None = None) -> Path:
    a = load_appearance(run)
    sel = good_crops(a)
    if max_frame is not None:
        sel &= a["frame"] <= max_frame
    idx = np.where(sel)[0]
    want = collections.defaultdict(list)                  # (cam, frame) -> sample indices
    for i in idx:
        want[(str(a["cam"][i]), int(a["frame"][i]))].append(i)
    sess = session()
    inp = sess.get_inputs()[0].name
    ds = run_dataset(run)
    emb = np.zeros((len(a["frame"]), 256), np.float16)
    done = np.zeros(len(a["frame"]), bool)
    buf, buf_i = [], []

    def flush():
        if buf:
            out = sess.run(None, {inp: np.stack(buf)})[0]
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            emb[buf_i] = out.astype(np.float16)
            done[buf_i] = True
            buf.clear()
            buf_i.clear()

    for cam_id, pos, _ in CAMERAS:
        frames = sorted(f for (c, f) in want if c == pos)
        if not frames:
            continue
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        targets, k, idx_f = set(frames), 0, 0
        last = frames[-1]
        while idx_f <= last:
            fr = idx_f              # kafka frame id = video frame index
            if fr in targets:
                ok, img = cap.read()
            else:
                ok, img = cap.grab(), None
            if not ok:
                break
            idx_f += 1
            if img is None:
                continue
            for i in want[(pos, fr)]:
                l, t, r, b = (int(v) for v in a["bbox"][i])
                crop = img[max(0, t):b, max(0, l):r]
                if crop.size == 0:
                    continue
                buf.append(preprocess(crop))
                buf_i.append(i)
                if len(buf) >= BATCH:
                    flush()
        flush()
        cap.release()
        print(f"{pos}: {int(done[a['cam'] == pos].sum())} embeddings", flush=True)
    out = run / "reid.npz"
    np.savez_compressed(out, frame=a["frame"][done], cam=a["cam"][done], gid=a["gid"][done], emb=emb[done])
    return out


def load(run: Path) -> dict:
    with np.load(run / "reid.npz") as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("run")
    ap.add_argument("--max-frame", type=int, default=None)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    print(extract(run, args.max_frame))


if __name__ == "__main__":
    main()
