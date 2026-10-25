"""Shot attribution: which player took each labelled shot, from the shot time (here the rim-motion labelling, in
deployment the 120 fps rim cameras) and player pose.

  prep   for every labelled shot (experiments/shots/shots_gt.json): three frames of the far camera facing that basket
         (FL for the left end, FR for the right end) at 1.5, 1.0 and 0.5 s before the ball reached the rim, every
         detection box drawn with its MV3DT object id, for labelling the shooter's box
         -> experiments/shots/attribution/<shot>.jpg, experiments/shots/attribution/boxes.json
  eval   the pose-based pick (the person near the basket with the highest both-arms-up score in the 2 s before the
         rim time, tools/shot_pose_eval.arms_up) against the labelled shooter box (experiments/shots/shooters.json)

  python -m tools.shot_attribution prep ct3_r1
  python -m tools.shot_attribution eval ct3_r1 [--model rtmpose-l]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import load_boxes, run_dataset

WIDE = {"left": "FL", "right": "FR"}
OFFSETS = (-45, -30, -15)
SHOTS = ROOT / "experiments" / "shots"


def prep(run: Path) -> None:
    shots = json.loads((SHOTS / "shots_gt.json").read_text())
    boxes = load_boxes(run, 1)
    ds = run_dataset(run)
    vid = {cam: ds / "videos" / f"{cid}_{cam}.mp4" for cid, cam, _ in CAMERAS}
    out = SHOTS / "attribution"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for s in shots:
        cam = WIDE[s["end"]]
        cap = cv2.VideoCapture(str(vid[cam].resolve()))
        tiles, ids = [], set()
        for off in OFFSETS:
            f = s["frame"] + off
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            for g, l, t, r, b in boxes.get((cam, f), []):
                if b - t < 40:
                    continue
                ids.add(str(g))
                cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 220, 255), 2)
                (tw, th), _ = cv2.getTextSize(str(g), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(img, (int(l), int(t) - th - 8), (int(l) + tw + 8, int(t)), (0, 220, 255), -1)
                cv2.putText(img, str(g), (int(l) + 4, int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(img, f"{cam}  {off / 30:+.1f} s before the rim", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (255, 255, 255), 3)
            tiles.append(cv2.resize(img, (1280, 720)))
        cv2.imwrite(str(out / f"{s['id']}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
        meta[s["id"]] = {"camera": cam, "frames": [s["frame"] + o for o in OFFSETS],
                         "ids": sorted(ids, key=int), **{k: s[k] for k in ("time", "end", "verdict", "shooter")}}
        print(s["id"], cam, s["time"], len(ids), "boxes", flush=True)
    (out / "boxes.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "eval"))
    ap.add_argument("run")
    ap.add_argument("--model", default="rtmpose-l")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_dir() else ROOT / "experiments" / a.run
    if a.cmd == "prep":
        prep(run)
    else:
        raise SystemExit("eval: needs experiments/shots/shooters.json (labelled shooter boxes)")


if __name__ == "__main__":
    main()
