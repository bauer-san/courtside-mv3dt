"""Cross-camera position disagreement, measured on the identity ground truth.

Every pair of labelled detections of the same person at the same moment from two different cameras (all court-filtered
detections of a moment are labelled, so the pairs do not depend on any tracker). Per detection two floor positions:
  mv3dt   MV3DT's footLocation3D (what the court tracker uses, minus its per-camera bias)
  bottom  the bbox bottom-centre back-projected onto z = 0
Per camera, the error of each detection against the mean of the other cameras' positions of that person is split into
radial (along the camera's line of sight on the floor; a foot-point / height problem) and tangential (across it;
a calibration problem), and related to box height, the box touching the image bottom, and distance to the camera.

  python -m tools.xcam <run> [--name game_2to3]
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.appearance import load_boxes
from tools.stitch import floor_point, load_cameras, load_detections
from tools.world_tracker import _by_frame, estimate_bias
from tools.evaluate import inside
from tools.stitch import Params as StitchParams, ground_filter


def labelled(name: str, run: Path | None = None) -> dict:
    """(frame, cam, gid) -> identity for every person label (team-only within a group counts; none/unclear do not).
    Team-only labels are made unique per group so they only pair with crops of the same group. Remapped to `run`'s
    detections if that is not the run the labels were made on (as gt_eval.load_gt)."""
    d = ROOT / "gt" / name
    tasks = json.loads((d / "tasks.json").read_text())
    out = {}
    for m in tasks["moments"]:
        for g in m["groups"]:
            p = d / "labels" / "labels" / f"{g['id']}.json"
            if not p.exists():
                continue
            lab = json.loads(p.read_text())
            for c in g["crops"]:
                i = lab.get("crops", {}).get(f"{c['cam']}:{c['gid']}") or lab["label"]
                if i in ("none", "unclear"):
                    continue
                if i.endswith(":?"):
                    i = f"{i}@{g['id']}"
                out[(m["frame"], c["cam"], str(c["gid"]))] = i
    src = ROOT / "experiments" / tasks["run"]
    if run is not None and Path(run).resolve() != src.resolve():
        from tools.gt_eval import remap
        out = remap(out, src, Path(run))
    return out


def camera_centre(P: np.ndarray) -> np.ndarray:
    return -np.linalg.solve(P[:, :3], P[:, 3])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--name", default="game_2to3")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    lab = labelled(args.name, run)
    cams = load_cameras(run / "camInfo_used")
    centre = {c: camera_centre(P) for c, P in cams.items()}
    d = load_detections(run / "kafka.jsonl.gz")
    keep = ground_filter(d, cams, inside, StitchParams(margin=0.3))
    raw, _ = _by_frame(d, keep)
    bias = estimate_bias(raw)
    print("tracker per-camera bias (m):", {c: tuple(round(v, 2) for v in b) for c, b in bias.items()})
    frames = sorted({k[0] for k in lab})
    boxes = load_boxes(run, frames[1] - frames[0])
    box = {(fr, cam, str(g)): (l, t, r, b) for (cam, fr), v in boxes.items() for g, l, t, r, b in v}
    idx = {(int(f), str(c), str(g)): i for i, (f, c, g) in enumerate(zip(d["frame"], d["cam"], d["gid"]))}

    rows = []      # one per labelled detection
    for k, ident in lab.items():
        i = idx.get(k)
        if i is None:
            continue
        l, t, r, b = box[k]
        cam = k[1]
        fp, ok = floor_point(cams[cam], np.array([(l + r) / 2]), np.array([b]))
        bx = bias.get(cam, (0.0, 0.0))
        rows.append(dict(frame=k[0], cam=cam, ident=ident, h=b - t, cut=b >= 1078,
                         mv3dt=np.array([d["x"][i] - bx[0], d["y"][i] - bx[1]]), mv3dt_raw=np.array([d["x"][i], d["y"][i]]),
                         bottom=fp[0]))
    people = collections.defaultdict(list)
    for r in rows:
        people[(r["frame"], r["ident"])].append(r)
    multi = [v for v in people.values() if len({r["cam"] for r in v}) > 1]
    print(f"{len(rows)} labelled detections, {len(multi)} person-moments seen by 2+ cameras")

    for est in ("mv3dt", "mv3dt_raw", "bottom"):
        pair = collections.defaultdict(list)
        for v in multi:
            for a in range(len(v)):
                for b_ in range(a + 1, len(v)):
                    if v[a]["cam"] != v[b_]["cam"]:
                        ca, cb = sorted((v[a]["cam"], v[b_]["cam"]))
                        ra, rb = (v[a], v[b_]) if v[a]["cam"] == ca else (v[b_], v[a])
                        pair[f"{ca}-{cb}"].append(np.linalg.norm(ra[est] - rb[est]))
        allp = [x for v in pair.values() for x in v]
        print(f"\n[{est}] pairwise distance between cameras: median {np.median(allp):.2f} m, "
              f"share > 0.9 m (merge distance) {np.mean(np.array(allp) > 0.9):.0%}, n={len(allp)}")
        for k in sorted(pair):
            v = np.array(pair[k])
            print(f"   {k}: n={len(v):3d}  median {np.median(v):.2f}  p75 {np.percentile(v, 75):.2f}  >0.9 m {np.mean(v > 0.9):.0%}")
        # per-camera error vs the other cameras' mean, radial / tangential to the camera
        per = collections.defaultdict(list)
        for v in multi:
            for r in v:
                others = [o[est] for o in v if o["cam"] != r["cam"]]
                ref = np.mean(others, axis=0)
                e = r[est] - ref
                c = centre[r["cam"]][:2]
                u = (ref - c) / (np.linalg.norm(ref - c) + 1e-9)
                per[r["cam"]].append((float(e @ u), float(e[0] * -u[1] + e[1] * u[0]), r["h"], r["cut"],
                                      float(np.linalg.norm(ref - c))))
        print(f"   per camera, error vs other cameras (radial + = farther from the camera than the others say):")
        for cam in sorted(per):
            a = np.array([x[:2] for x in per[cam]])
            h = np.array([x[2] for x in per[cam]])
            dist = np.array([x[4] for x in per[cam]])
            cut = np.array([x[3] for x in per[cam]])
            cr = np.corrcoef(a[:, 0], dist)[0, 1] if len(a) > 2 else np.nan
            print(f"     {cam}: n={len(a):3d}  radial mean {a[:, 0].mean():+.2f} (sd {a[:, 0].std():.2f})  "
                  f"tangential mean {a[:, 1].mean():+.2f} (sd {a[:, 1].std():.2f})  "
                  f"corr(radial, distance) {cr:+.2f}  cut boxes {int(cut.sum())}")


if __name__ == "__main__":
    main()
