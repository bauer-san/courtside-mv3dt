"""Body keypoints for every detection of a run (NVIDIA BodyPose3DNet, the pose model MV3DT bundles).

For every `step`-th frame and every recorded detection (kafka.jsonl.gz boxes) at least `min_h` px tall, the box is
cut out, stretched to 192 x 256 and scaled to [0, 1] in BGR order, and the model's 34 keypoints are mapped back to
image pixels (with their confidences). Preprocessing and constant inputs follow NVIDIA's deepstream-bodypose-3d app
(configs/config_infer_secondary_bodypose3dnet.txt, nvdsinitinputlayers_BodyPose3DNet.cpp).

Keypoints: 0 pelvis, 1/2 hips, 3 torso, 4/5 knees, 6 neck, 7/8 ankles, 9-14 feet, 15 nose, 16/17 eyes, 18/19 ears,
20/21 shoulders, 22/23 elbows, 24/25 wrists, 26-33 hands (left/right pairs).

  python -m tools.pose_extract ct3_r1 [--end 12600] [--step 2] -> experiments/<run>/pose.npz
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


def session():
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    return ort.InferenceSession(str(MODEL), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def infer(s, crops: list[np.ndarray]) -> np.ndarray:
    n = len(crops)
    return s.run(["pose2d"], {"input0": np.stack(crops), "k_inv": np.repeat(KINV[None], n, 0),
                              "t_form_inv": np.repeat(np.eye(3, dtype=np.float32)[None], n, 0),
                              "scale_normalized_mean_limb_lengths": np.repeat(SN[None], n, 0),
                              "mean_limb_lengths": np.repeat(ML[None], n, 0)})[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--end", type=int, default=12600)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min-h", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_dir() else ROOT / "experiments" / a.run
    boxes = load_boxes(run, a.step)
    ds = run_dataset(run)
    s = session()
    rows = {k: [] for k in ("frame", "cam", "gid", "box", "kp")}
    for cid, cam, _ in CAMERAS:
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cid}_{cam}.mp4").resolve()))
        crops, meta = [], []

        def flush():
            if not crops:
                return
            out = infer(s, crops)
            for p, (f, g, l, t, r, b) in zip(out, meta):
                kp = p.copy()
                kp[:, 0] = l + kp[:, 0] * (r - l) / 192
                kp[:, 1] = t + kp[:, 1] * (b - t) / 256
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
                crops.append((cv2.resize(img[t:b, l:r], (192, 256)).astype(np.float32) / 255).transpose(2, 0, 1))
                meta.append((f, str(g), l, t, r, b))
                if len(crops) >= a.batch:
                    flush()
        flush()
        print(cam, sum(1 for c in rows["cam"] if c == cam), "detections", flush=True)
    out = run / "pose.npz"
    np.savez_compressed(out, frame=np.array(rows["frame"]), cam=np.array(rows["cam"]), gid=np.array(rows["gid"]),
                        box=np.array(rows["box"], np.int16), kp=np.array(rows["kp"]))
    print(out)


if __name__ == "__main__":
    main()
