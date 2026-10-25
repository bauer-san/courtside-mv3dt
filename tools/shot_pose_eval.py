"""Can player pose find shot attempts? Arms-up events (tools/pose_extract.py keypoints) against the labelled shots.

Per detection, the arms-up score is how far the wrists are above the shoulders, in torso lengths (neck to pelvis in
the image), from keypoints with confidence >= --conf: `both` takes the lower wrist (both arms up), `one` the higher.
A detection is flagged when its score reaches --thr and its floor point (MV3DT foot location) is within --range m of
a basket; flags within --gap s at the same basket form one event. A labelled shot (experiments/shots/shots_gt.json,
from the rim-motion labelling) is found when an event at its basket falls between --pre s and 0.15 s before the ball
reached the rim. Every other event is a false alarm.

  python -m tools.shot_pose_eval ct3_r1 [--mode both|one] [--thr 0.3]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.court import FT, HALF_LEFT_FT, HALF_RIGHT_FT

BASKET = {"left": np.array([-(HALF_LEFT_FT - 5.25) * FT, 0.0]), "right": np.array([(HALF_RIGHT_FT - 5.25) * FT, 0.0])}


LAYOUT = {   # keypoint indices: (left, right) shoulders and wrists, torso top and bottom (single or pairs)
    "bodypose3d": dict(sh=(20, 21), wr=(24, 25), top=(6,), bottom=(0,)),
    "coco": dict(sh=(5, 6), wr=(9, 10), top=(5, 6), bottom=(11, 12)),
}


def arms_up(kp: np.ndarray, conf: float, layout: str = "bodypose3d") -> tuple[np.ndarray, np.ndarray]:
    """(both, one) arms-up scores per detection; NaN where the keypoints are not confident."""
    L = LAYOUT[layout]
    mid = lambda idx: kp[:, list(idx), :2].mean(1)
    sure = lambda idx: kp[:, list(idx), 2].min(1) >= conf
    torso = np.hypot(*(mid(L["top"]) - mid(L["bottom"])).T)
    ok_t = sure(L["top"]) & sure(L["bottom"]) & (torso > 5)
    sc = []
    for w, sh in zip(L["wr"], L["sh"]):
        up = (kp[:, sh, 1] - kp[:, w, 1]) / np.where(torso > 0, torso, 1)
        sc.append(np.where(ok_t & (kp[:, w, 2] >= conf) & (kp[:, sh, 2] >= conf), up, np.nan))
    a, b = sc
    both = np.where(np.isnan(a) | np.isnan(b), np.nan, np.minimum(a, b))   # needs both arms confident
    return both, np.fmax(a, b)                                              # either arm


def events(frames: np.ndarray, gap: int) -> list[tuple[int, int]]:
    ev = []
    for f in np.sort(np.unique(frames)):
        if ev and f - ev[-1][1] <= gap:
            ev[-1][1] = f
        else:
            ev.append([f, f])
    return [tuple(e) for e in ev]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--model", choices=("bodypose3d", "rtmpose-m", "rtmpose-l"), default="bodypose3d")
    ap.add_argument("--mode", choices=("both", "one"), default="both")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--range", type=float, default=9.0)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--end", type=int, default=12600)
    ap.add_argument("--sweep", action="store_true", help="table over modes and thresholds")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_dir() else ROOT / "experiments" / a.run
    z = np.load(run / ("pose.npz" if a.model == "bodypose3d" else f"pose_{a.model}.npz"))
    from tools.world_tracker import load_detections
    d = load_detections(run / "kafka.jsonl.gz")
    pos = {(int(f), str(c), str(g)): (x, y) for f, c, g, x, y in zip(d["frame"], d["cam"], d["gid"], d["x"], d["y"])}
    xy = np.array([pos.get((int(f), str(c), str(g)), (np.nan, np.nan)) for f, c, g in zip(z["frame"], z["cam"], z["gid"])])
    shots = json.loads((ROOT / "experiments" / "shots" / "shots_gt.json").read_text())
    both, one = arms_up(z["kp"], a.conf, "bodypose3d" if a.model == "bodypose3d" else "coco")
    minutes = a.end / 30 / 60

    def evaluate(score, thr):
        found, fa = 0, 0
        hits = []
        for end, B in BASKET.items():
            near = np.hypot(*(xy - B).T) <= a.range
            fr = z["frame"][(score >= thr) & near & (z["frame"] <= a.end)]
            evs = events(fr, int(a.gap * 30))
            sh = [s for s in shots if s["end"] == end]
            used = set()
            for s in sh:
                lo, hi = s["frame"] - a.pre * 30, s["frame"] - 5
                m = [i for i, (e0, e1) in enumerate(evs) if e1 >= lo and e0 <= hi]
                if m:
                    found += 1
                    used.update(m)
                    hits.append(s["id"])
            fa += len(evs) - len(used)
        return found, fa, hits

    if a.sweep:
        print(f"| arms-up rule | threshold (torso lengths) | shots found (of {len(shots)}) | false alarms per minute |")
        print("|---|---|---|---|")
        for mode, sc in (("both wrists above shoulders", both), ("one wrist above shoulder", one)):
            for thr in (0.0, 0.2, 0.4, 0.6, 0.8):
                f, fa, _ = evaluate(sc, thr)
                print(f"| {mode} | {thr:.1f} | {f} | {fa / minutes:.1f} |")
        return
    f, fa, hits = evaluate(both if a.mode == "both" else one, a.thr)
    print(f"{a.mode} >= {a.thr}: shots found {f}/{len(shots)} ({', '.join(hits)}); false alarms {fa} "
          f"({fa / minutes:.1f} per minute)")
    valid = np.isfinite(both)
    print(f"detections {len(both)}, with confident arm keypoints {valid.mean():.0%}")


if __name__ == "__main__":
    main()
