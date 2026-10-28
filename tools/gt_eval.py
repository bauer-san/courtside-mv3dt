"""Score trackers against the small identity ground truth (tools/gt_prepare.py + the labelling page).

Labels: gt/<name>/labels/labels/<group id>.json (the page's `labels` collection, saved with ArtifactData
out_dir). A detection's identity is its crop override, else its group's label. Only full identities count
(red:7, maroon:94, ref:A); team-only (red:?), not-a-player and unclear detections are left out.

Per tracker, over the labelled moments (2 s apart):
  switches      consecutive moments of one player where the track holding them changed (per player, summed)
  tracks/player distinct tracks that held the player
  impure tracks tracks holding more than one player (identity swaps or merges)
  duplicates    moments where one player was held by two tracks at once
  missed        labelled detections no track held
  IDF1          share of labelled detections on the track matched one-to-one to their player (Hungarian on
                shared detections; a player's other tracks and a track's other players count as errors)

  python -m tools.gt_eval <run> [--name game_2to3] [--md out.md]
"""

import argparse
import collections
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from tools import ROOT
from tools.final_eval import mv3dt_tracks
from tools.world_tracker import WTParams, identity_tracks, track_run


_BOXES = {}


def _boxes(run: Path, step: int) -> dict:
    from tools.appearance import load_boxes
    if (run, step) not in _BOXES:
        _BOXES[(run, step)] = load_boxes(run, step)
    return _BOXES[(run, step)]


def frame_offsets(run: Path) -> dict:
    """Camera position -> first source frame of the run's dataset clip for that camera, relative to the clip start
    (tools/make_clips --offsets; empty when the cameras' clips start at the same source frame)."""
    from tools.appearance import run_dataset
    m = run_dataset(run) / "manifest.json"
    return json.loads(m.read_text()).get("offsets", {}) if m.exists() else {}


def game_end(run: Path) -> int:
    """Last game frame of the run's clips (tools.GAME_END is in the original NL frames; a time-aligned dataset's clips
    start later by their NL offset)."""
    from tools import GAME_END
    return GAME_END - frame_offsets(run).get("NL", 0)


def remap(gt: dict, src: Path, dst: Path, step: int = 60, min_iou: float = 0.6) -> dict:
    """Carry labels made on run `src`'s detections over to run `dst` (other MV3DT ids): same moment and camera (the
    frame shifted by the datasets' per-camera offsets, if any), best box overlap >= min_iou."""
    off_src, off_dst = frame_offsets(src), frame_offsets(dst)
    a = _boxes(src, step)
    b = _boxes(dst, 1 if off_src != off_dst else step)
    out = {}
    for (fr, cam, gid), ident in gt.items():
        box = next((x[1:] for x in a.get((cam, fr), []) if str(x[0]) == gid), None)
        if box is None:
            continue
        fr_dst = fr + off_src.get(cam, 0) - off_dst.get(cam, 0)
        best, best_iou = None, min_iou
        for g2, l, t, r, bb in b.get((cam, fr_dst), []):
            iw = min(r, box[2]) - max(l, box[0])
            ih = min(bb, box[3]) - max(t, box[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            iou = inter / ((r - l) * (bb - t) + (box[2] - box[0]) * (box[3] - box[1]) - inter)
            if iou > best_iou:
                best, best_iou = str(g2), iou
        if best is not None:
            out[(fr_dst, cam, best)] = ident
    return out


def load_gt(name: str, run: Path | None = None) -> dict:
    """(frame, cam, gid) -> identity, full identities only; remapped to `run`'s detections if that is not the run
    the labels were made on."""
    d = ROOT / "gt" / name
    tasks = json.loads((d / "tasks.json").read_text())
    src = ROOT / "experiments" / tasks["run"]
    labels = {p.stem: json.loads(p.read_text()) for p in (d / "labels" / "labels").glob("*.json")}
    gt = {}
    for m in tasks["moments"]:
        for g in m["groups"]:
            lab = labels.get(g["id"])
            if not lab or not lab.get("label"):
                continue
            body = lab.get("data", lab)          # ArtifactData files may wrap the fields
            for c in g["crops"]:
                ident = body.get("crops", {}).get(f"{c['cam']}:{c['gid']}") or body["label"]
                if ":" in ident and not ident.endswith(":?"):
                    gt[(m["frame"], c["cam"], str(c["gid"]))] = ident
    if run is not None and Path(run).resolve() != src.resolve():
        gt = remap(gt, src, Path(run))
    return gt


def score(tracks: list, gt: dict) -> dict:
    owner = {}
    for i, t in enumerate(tracks):
        for k in t.dets:
            k = (int(k[0]), str(k[1]), str(k[2]))
            if k in gt:
                owner[k] = i
    per = collections.defaultdict(lambda: collections.defaultdict(set))   # player -> frame -> track ids
    missed = 0
    for k, ident in gt.items():
        if k in owner:
            per[ident][k[0]].add(owner[k])
        else:
            missed += 1
    switches = dup = 0
    tracks_of = {}
    for ident, frames in per.items():
        prev = None
        for f in sorted(frames):
            ids = frames[f]
            dup += len(ids) > 1
            if prev is not None and not (ids & prev):
                switches += 1
            prev = ids
        tracks_of[ident] = set().union(*frames.values())
    holders = collections.defaultdict(set)
    for ident, ts in tracks_of.items():
        for t in ts:
            holders[t].add(ident)
    n_players = len(per)
    players, tids = sorted(per), sorted(holders)
    C = np.zeros((len(players), len(tids)))
    pi, ti = {p: i for i, p in enumerate(players)}, {t: i for i, t in enumerate(tids)}
    for k, ident in gt.items():
        if k in owner:
            C[pi[ident], ti[owner[k]]] += 1
    r, c = linear_sum_assignment(-C)
    idf1 = C[r, c].sum() / max(len(gt), 1)
    return dict(idf1=idf1, players=n_players, labelled=len(gt), switches=switches,
                tracks_per_player=sum(len(v) for v in tracks_of.values()) / max(n_players, 1),
                impure=sum(len(v) > 1 for v in holders.values()), tracks=len(holders), duplicates=dup, missed=missed)


COLS = [("variant", "{}"), ("idf1", "{:.0%}"), ("switches", "{}"), ("tracks_per_player", "{:.1f}"), ("impure", "{}"), ("tracks", "{}"),
        ("duplicates", "{}"), ("missed", "{}")]
HEAD = ["variant", "IDF1", "switches", "tracks/player", "impure tracks", "tracks", "duplicates", "missed dets"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--name", default="game_2to3")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    gt = load_gt(args.name, run)
    print(f"{len(gt)} labelled detections on {run.name}")
    variants = [
        ("MV3DT global IDs (court-filtered)", None),
        ("court tracker, no cues", dict(p=WTParams(online_relink=True, use_numbers=False), appearance=False)),
        ("+ team colour", dict(p=WTParams(online_relink=True, use_numbers=False))),
        ("+ team colour + numbers", dict(p=WTParams(online_relink=True))),
        ("+ team colour + numbers, IDENTITY OUTPUT (online roster, per-player look)", dict(p=WTParams(online_relink=True), ident=True)),
        ("+ team colour + numbers, old rule (team optional, 60 s relink)",
         dict(p=WTParams(online_relink=True, num_needs_team=False, id_gap=1800))),
        ("old association (isotropic, plain Mahalanobis) + team colour",
         dict(p=WTParams(online_relink=True, use_numbers=False, aniso=False, cost_logdet=False))),
    ]
    rows = []
    for name, kw in variants:
        if kw is None:
            T = mv3dt_tracks(run)
        elif kw.pop("ident", False):
            T = identity_tracks(track_run(run, **kw), track_run.last_det_ident)
        else:
            T = track_run(run, **kw)
        m = score(T, gt)
        m["variant"] = name
        rows.append(m)
        print(" | ".join(fmt.format(m[c]) for c, fmt in COLS), flush=True)
    md = (f"### Ground truth {args.name} ({rows[0]['labelled']} labelled detections, {rows[0]['players']} people)\n\n"
          "| " + " | ".join(HEAD) + " |\n|" + "---|" * len(HEAD) + "\n"
          + "".join("| " + " | ".join(fmt.format(r[c]) for c, fmt in COLS) + " |\n" for r in rows))
    print(md)
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
