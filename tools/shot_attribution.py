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


def evaluate(run: Path, model: str) -> None:
    from tools.shot_pose_eval import BASKET, arms_up
    from tools.world_tracker import load_detections
    shots = {s["id"]: s for s in json.loads((SHOTS / "shots_gt.json").read_text())}
    meta = json.loads((SHOTS / "attribution" / "boxes.json").read_text())
    truth = json.loads((SHOTS / "shooters.json").read_text())
    z = np.load(run / ("pose.npz" if model == "bodypose3d" else f"pose_{model}.npz"))
    d = load_detections(run / "kafka.jsonl.gz")
    pos = {(int(f), str(c), str(g)): np.array([x, y]) for f, c, g, x, y in zip(d["frame"], d["cam"], d["gid"], d["x"], d["y"])}
    xy = np.array([pos.get((int(f), str(c), str(g)), (np.nan, np.nan)) for f, c, g in zip(z["frame"], z["cam"], z["gid"])])
    both, one = arms_up(z["kp"], 0.3, "bodypose3d" if model == "bodypose3d" else "coco")
    rows = []
    for sid, box in truth.items():
        s, m = shots[sid], meta[sid]
        win = (z["frame"] >= s["frame"] - 60) & (z["frame"] <= s["frame"] - 5) & \
            (np.hypot(*(xy - BASKET[s["end"]]).T) <= 9)
        res = {"id": sid, "time": s["time"], "truth": box}
        for name, sc in (("both", both), ("one", one)):
            cand = np.where(win & np.isfinite(sc))[0]
            if not len(cand):
                res[name] = "no candidate"
                continue
            i = cand[np.argmax(sc[cand])]
            if box in ("none", "unsure"):
                res[name] = "-"
                continue
            f, cam, g = int(z["frame"][i]), str(z["cam"][i]), str(z["gid"][i])
            if cam == m["camera"] and g == box:
                res[name] = "correct"
                continue
            # another camera, or another id: compare court positions with the labelled shooter at the nearest frame
            near = [pos[(ff, m["camera"], box)] for ff in range(f - 6, f + 7) if (ff, m["camera"], box) in pos]
            p = pos.get((f, cam, g))
            if near and p is not None:
                dist = min(np.hypot(*(p - q)) for q in near)
                res[name] = "correct" if dist < 1.0 else f"wrong ({dist:.1f} m away)"
            else:
                res[name] = "wrong (no position)"
        rows.append(res)
    print("| shot | time | labelled box | pick: both arms up | pick: highest wrist |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['time']} | {r['truth']} | {r['both']} | {r['one']} |")
    lab = [r for r in rows if r["truth"] not in ("none", "unsure")]
    for name in ("both", "one"):
        ok = sum(r[name] == "correct" for r in lab)
        print(f"{name}: correct {ok} of {len(lab)} shots with a visible shooter")


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
        evaluate(run, a.model)


if __name__ == "__main__":
    main()
