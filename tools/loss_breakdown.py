"""Where identity IDF1 is lost, over several DeepStream runs and both labelled minutes.

Each labelled detection is classified for the identity output (tools/world_tracker.identity_tracks):
  correct              on the output unit matched (one-to-one, Hungarian) to its player
  missed               no court track holds the detection
  unidentified         on a track without an identity at that frame that is not the player's matched unit;
                       split by whether the track had readable crops (box >= 150 px, not cut) and jersey reads
  wrong team           reported as the right number on the other team
  wrong number, carried  reported as another identity that the same track held for another player earlier
                         (the track swapped players and kept the identity)
  wrong number, other  any other wrong identity
  other                remaining (e.g. on an identity matched to another player while its own is unmatched)

  python -m tools.loss_breakdown base_r1 base_r2 base_r3 [--md out.md]
"""

import argparse
import collections
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools import ROOT
from tools.appearance import load_appearance
from tools.gt_eval import load_gt
from tools.world_tracker import WTParams, track_run

GT = ("game_2to3", "game_4to5")


def readable_keys(run: Path) -> set:
    a = load_appearance(run)
    h = a["bbox"][:, 3] - a["bbox"][:, 1]
    sel = (~a["cut"]) & (h >= 150)
    return {(int(f), str(c), str(g)) for f, c, g in zip(a["frame"][sel], a["cam"][sel], a["gid"][sel])}


def breakdown(run: Path, gts: dict) -> dict:
    T = track_run(run, WTParams(online_relink=True))
    di = track_run.last_det_ident
    readable = readable_keys(run)
    owner, label = {}, {}
    for t in T:
        for k in t.dets:
            k = (int(k[0]), str(k[1]), str(k[2]))
            owner[k] = t
            label[k] = di.get(k) or f"track{t.tid}"
    out = {}
    for name, gt in gts.items():
        C = collections.Counter((gt[k], label[k]) for k in gt if k in label)
        ids = sorted({a for a, _ in C})
        labs = sorted({b for _, b in C})
        M = np.zeros((len(ids), len(labs)))
        for (a, b), v in C.items():
            M[ids.index(a), labs.index(b)] = v
        r, c = linear_sum_assignment(-M)
        match = {ids[i]: labs[j] for i, j in zip(r, c) if M[i, j] > 0}
        # identities each track showed for each true player (for "carried" detection)
        seen = collections.defaultdict(set)              # (track id, shown identity) -> true players
        for k, true in gt.items():
            if k in label and ":" in label[k]:
                seen[(id(owner[k]), label[k])].add(true)
        cnt = collections.Counter()
        for k, true in gt.items():
            if k not in label:
                cnt["missed"] += 1
                continue
            lab = label[k]
            if match.get(true) == lab:
                cnt["correct"] += 1
            elif lab.startswith("track"):
                t = owner[k]
                n_read = sum(1 for d in t.dets if (int(d[0]), str(d[1]), str(d[2])) in readable)
                kind = ("unidentified: readable crops, no usable reads" if n_read and not t.reads else
                        "unidentified: readable crops and reads, identity not established" if n_read else
                        "unidentified: no readable crops")
                cnt[kind] += 1
            else:
                tt, tn = true.split(":")
                lt, ln = lab.split(":")
                if ln == tn and lt != tt:
                    cnt["wrong team (right number)"] += 1
                elif any(p != true for p in seen[(id(owner[k]), lab)]) and lab in [match.get(p) for p in
                                                                                 seen[(id(owner[k]), lab)]]:
                    cnt["wrong number, carried from another player"] += 1
                elif ln != tn:
                    cnt["wrong number, other"] += 1
                else:
                    cnt["other"] += 1
        out[name] = (cnt, len(gt))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    runs = [Path(r) if Path(r).is_dir() else ROOT / "experiments" / r for r in args.runs]
    rows = collections.defaultdict(list)
    for run in runs:
        gts = {n: load_gt(n, run) for n in GT}
        for name, (cnt, n) in breakdown(run, gts).items():
            print(f"{run.name} {name}: " + ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])),
                  flush=True)
            for k in set(cnt) | set(rows):
                rows[k].append(cnt.get(k, 0) / n)
    order = sorted(rows, key=lambda k: -np.mean(rows[k]))
    lines = [f"### Identity IDF1 loss over {len(runs)} runs x {len(GT)} minutes (share of labelled detections)", "",
             "| category | mean | range |", "|---|---|---|"]
    for k in order:
        v = np.array(rows[k] + [0.0] * (len(runs) * len(GT) - len(rows[k])))
        lines.append(f"| {k} | {v.mean():.1%} | {v.min():.0%}-{v.max():.0%} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
