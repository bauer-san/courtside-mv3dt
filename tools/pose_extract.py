"""Body keypoints for every detection of a run, for the pose-based shot test (tools/shot_pose_eval.py).

For every `step`-th frame and every recorded detection (kafka.jsonl.gz boxes) at least `min_h` px tall, the person
crop goes through a pose model and its keypoints are mapped back to image pixels, with their confidences.

  bodypose3d  NVIDIA BodyPose3DNet (bundled with MV3DT): box stretched to 192 x 256, [0, 1] BGR, as NVIDIA's
              deepstream-bodypose-3d app. 34 keypoints: 0 pelvis, 1/2 hips, 3 torso, 4/5 knees, 6 neck, 7/8 ankles,
              9-14 feet, 15 nose, 16/17 eyes, 18/19 ears, 20/21 shoulders, 22/23 elbows, 24/25 wrists, 26-33 hands.
  rtmpose-m / rtmpose-l   RTMPose body7 256 x 192 (OpenMMLab, Apache-2.0; models/rtmpose/): box padded 1.25x at a
              3:4 aspect, RGB normalised with ImageNet mean/std, SimCC outputs (argmax / 2, confidence = the smaller
              of the x and y maxima), as the model's pipeline.json. 17 COCO keypoints: 0 nose, 1/2 eyes, 3/4 ears,
              5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.

  python -m tools.pose_extract ct3_r1 [--model rtmpose-m] [--end 12600] [--step 2] -> experiments/<run>/pose_<model>.npz
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import load_boxes, run_dataset

MODEL = (ROOT / "external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/BodyPose3DNet/"
         "bodypose3dnet_accuracy.onnx")
SN = np.array([0.5, 0.5, 1.0, 0.8175, 0.9889, 0.2610, 0.7942, 0.5724, 0.5078, 0, 0, 0, 0, 0, 0, 0, 0.3433, 0.8171,
               0.9912, 0.2610, 0.8259, 0.5724, 0.5078, 0, 0, 0, 0, 0, 0, 0, 0.3422, 0, 0, 0, 0, 0], np.float32)
ML = np.array([246.3427, 246.3427, 492.6854, 402.4380, 487.0321, 128.6856, 391.6295, 281.9928, 249.9478, 0, 0, 0, 0,
               0, 0, 0, 169.1832, 402.2611, 488.1824, 128.6848, 407.5836, 281.9897, 249.9489, 0, 0, 0, 0, 0, 0, 0,
               168.6137, 0, 0, 0, 0, 0], np.float32)
KINV = np.array([[0.00124876620338, 0, -0.119881555525], [0, 0.00124876620338, -0.159842074033], [0, 0, 1]], np.float32)


RTMPOSE = {m: next((ROOT / "models" / "rtmpose" / m[-1]).rglob("end2end.onnx"), None) for m in ("rtmpose-m", "rtmpose-l")}
MEAN, STD = np.array([123.675, 116.28, 103.53], np.float32), np.array([58.395, 57.12, 57.375], np.float32)


def session(model: str):
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    path = MODEL if model == "bodypose3d" else RTMPOSE[model]
    return ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def crop(model: str, img: np.ndarray, l: int, t: int, r: int, b: int):
    """Network input for one box, and the (x0, y0, sx, sy) that maps network pixels back to the image."""
    if model == "bodypose3d":
        x = (cv2.resize(img[t:b, l:r], (192, 256)).astype(np.float32) / 255).transpose(2, 0, 1)
        return x, (l, t, (r - l) / 192, (b - t) / 256)
    cx, cy, w, h = (l + r) / 2, (t + b) / 2, (r - l) * 1.25, (b - t) * 1.25
    if w > h * 0.75:
        h = w / 0.75
    else:
        w = h * 0.75
    x0, y0 = cx - w / 2, cy - h / 2
    M = np.array([[192 / w, 0, -x0 * 192 / w], [0, 256 / h, -y0 * 256 / h]], np.float32)
    patch = cv2.warpAffine(img, M, (192, 256), flags=cv2.INTER_LINEAR)
    x = ((cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
    return x, (x0, y0, w / 192, h / 256)


def infer(model: str, s, crops: list[np.ndarray]) -> np.ndarray:
    """(n, K, 3) keypoints in network pixels with confidences."""
    n = len(crops)
    if model == "bodypose3d":
        return s.run(["pose2d"], {"input0": np.stack(crops), "k_inv": np.repeat(KINV[None], n, 0),
                                  "t_form_inv": np.repeat(np.eye(3, dtype=np.float32)[None], n, 0),
                                  "scale_normalized_mean_limb_lengths": np.repeat(SN[None], n, 0),
                                  "mean_limb_lengths": np.repeat(ML[None], n, 0)})[0]
    sx, sy = s.run(None, {"input": np.stack(crops)})
    out = np.zeros((n, sx.shape[1], 3), np.float32)
    out[..., 0] = sx.argmax(-1) / 2.0
    out[..., 1] = sy.argmax(-1) / 2.0
    out[..., 2] = np.minimum(sx.max(-1), sy.max(-1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--model", choices=("bodypose3d", "rtmpose-m", "rtmpose-l"), default="bodypose3d")
    ap.add_argument("--end", type=int, default=12600)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min-h", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_dir() else ROOT / "experiments" / a.run
    boxes = load_boxes(run, a.step)
    ds = run_dataset(run)
    s = session(a.model)
    rows = {k: [] for k in ("frame", "cam", "gid", "box", "kp")}
    for cid, cam, _ in CAMERAS:
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cid}_{cam}.mp4").resolve()))
        crops, meta = [], []

        def flush():
            if not crops:
                return
            out = infer(a.model, s, crops)
            for p, (f, g, l, t, r, b, (x0, y0, sx, sy)) in zip(out, meta):
                kp = p.copy()
                kp[:, 0] = x0 + kp[:, 0] * sx
                kp[:, 1] = y0 + kp[:, 1] * sy
                rows["frame"].append(f); rows["cam"].append(cam); rows["gid"].append(g)
                rows["box"].append((l, t, r, b)); rows["kp"].append(kp.astype(np.float32))
            crops.clear(); meta.clear()

        for f in range(a.end + 1):
            if f % a.step or (cam, f) not in boxes:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            for g, l, t, r, b in boxes[(cam, f)]:
                l, t, r, b = int(max(0, l)), int(max(0, t)), int(min(img.shape[1] - 1, r)), int(min(img.shape[0] - 1, b))
                if b - t < a.min_h or r - l < 8:
                    continue
                x, back = crop(a.model, img, l, t, r, b)
                crops.append(x)
                meta.append((f, str(g), l, t, r, b, back))
                if len(crops) >= a.batch:
                    flush()
        flush()
        print(cam, sum(1 for c in rows["cam"] if c == cam), "detections", flush=True)
    out = run / ("pose.npz" if a.model == "bodypose3d" else f"pose_{a.model}.npz")
    np.savez_compressed(out, frame=np.array(rows["frame"]), cam=np.array(rows["cam"]), gid=np.array(rows["gid"]),
                        box=np.array(rows["box"], np.int16), kp=np.array(rows["kp"]), model=a.model)
    print(out)


if __name__ == "__main__":
    main()
