"""Why identities break: classify every ground-truth switch and duplicate of a court-tracker variant.

A switch: player P is held by track(s) S1 at one labelled moment and by disjoint track(s) S2 at the next.
With A = the track that held P before and B = the one that holds P after:
  swap            A is alive at the second moment and holds another labelled player
  duplicate       A is alive and within DUP_DIST of P, but the detection went to B (two tracks on one player)
  drifted         A is alive but elsewhere (on an unlabelled detection or coasting away from P)
  takeover-swap   A ended; B already existed and held another labelled player at the first moment
  takeover-dup    A ended; B already existed (a parallel track on P or an unlabelled detection)
  lost/reborn     A ended (coast / split / merged); B started after it and was not re-linked
For each switch it also notes whether P's per-camera tracklet (camera + MV3DT id) was the same at both
moments: then the per-camera tracker followed P continuously and the break is in the court tracker.
A duplicate moment: P is held by two tracks at once; split by whether the tracks' detections come from the same
camera (a double detection) or different cameras (cross-camera fusion failure), with their distance.

  python -m tools.gt_causes <run> [--name game_2to3] [--md out.md]
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.gt_eval import load_gt
from tools.world_tracker import WTParams, track_run

DUP_DIST = 1.5   # m
CAUSES = ["swap", "duplicate", "drifted", "takeover-swap", "takeover-dup", "lost/reborn"]


def pos_at(t, f: int, tol: int = 15):
    best = min(t.out, key=lambda o: abs(o[0] - f))
    return np.array(best[1:3]) if abs(best[0] - f) <= tol else None


def analyse(tracks: list, gt: dict) -> dict:
    owner = {}
    for i, t in enumerate(tracks):
        for k in t.dets:
            k = (int(k[0]), str(k[1]), str(k[2]))
            if k in gt:
                owner[k] = i
    per = collections.defaultdict(lambda: collections.defaultdict(set))     # player -> frame -> tracks
    keys = collections.defaultdict(lambda: collections.defaultdict(list))   # player -> frame -> det keys
    holds = collections.defaultdict(set)                                    # (track, frame) -> players
    for k, ident in gt.items():
        if k in owner:
            per[ident][k[0]].add(owner[k])
            keys[ident][k[0]].append(k)
            holds[(owner[k], k[0])].add(ident)
    first = [t.out[0][0] for t in tracks]
    last = [t.out[-1][0] for t in tracks]
    events, dups = [], []
    for P, frames in per.items():
        fs = sorted(frames)
        for f in fs:
            if len(frames[f]) > 1:
                ts = sorted(frames[f])
                cams = [{k[1] for k in keys[P][f] if owner[k] == i} for i in ts]
                same = any(cams[a] & cams[b] for a in range(len(ts)) for b in range(a + 1, len(ts)))
                ps = [pos_at(tracks[i], f) for i in ts]
                d = max((np.linalg.norm(ps[a] - ps[b]) for a in range(len(ts)) for b in range(a + 1, len(ts))
                         if ps[a] is not None and ps[b] is not None), default=np.nan)
                dups.append(dict(player=P, frame=f, same_cam=same, dist=d, cams=["+".join(sorted(c)) for c in cams]))
        for f1, f2 in zip(fs, fs[1:]):
            S1, S2 = frames[f1], frames[f2]
            if S1 & S2:
                continue
            B = min(S2, key=lambda i: first[i])
            alive = [a for a in S1 if first[a] <= f2 <= last[a]]
            A = alive[0] if alive else max(S1, key=lambda i: last[i])
            tl1 = {(k[1], k[2]) for k in keys[P][f1]}
            tl2 = {(k[1], k[2]) for k in keys[P][f2]}
            ev = dict(player=P, f1=f1, f2=f2, A=tracks[A].tid, B=tracks[B].tid, same_tracklet=bool(tl1 & tl2))
            if alive:
                other = holds.get((A, f2), set()) - {P}
                pa, pb = pos_at(tracks[A], f2), pos_at(tracks[B], f2)
                d = float(np.linalg.norm(pa - pb)) if pa is not None and pb is not None else np.nan
                ev["dist"] = d
                if other:
                    ev["cause"], ev["other"] = "swap", ",".join(sorted(other))
                elif d < DUP_DIST:
                    ev["cause"] = "duplicate"
                else:
                    ev["cause"] = "drifted"
            else:
                ev["a_end"] = tracks[A].end or "?"
                if first[B] <= last[A]:
                    other = holds.get((B, f1), set()) - {P}
                    ev["cause"] = "takeover-swap" if other else "takeover-dup"
                    if other:
                        ev["other"] = ",".join(sorted(other))
                else:
                    ev["cause"] = "lost/reborn"
                    ev["gap"] = first[B] - last[A]
                    pa, pb = np.array(tracks[A].out[-1][1:3]), np.array(tracks[B].out[0][1:3])
                    ev["dist"] = float(np.linalg.norm(pa - pb))
            events.append(ev)
    return dict(events=events, dups=dups, players=len(per))


def summary(name: str, r: dict) -> str:
    ev, dups = r["events"], r["dups"]
    c = collections.Counter(e["cause"] for e in ev)
    cont = collections.Counter(e["cause"] for e in ev if e["same_tracklet"])
    ends = collections.Counter(e.get("a_end") for e in ev if e["cause"] == "lost/reborn")
    gaps = [e["gap"] for e in ev if e["cause"] == "lost/reborn"]
    dists = [e["dist"] for e in ev if e["cause"] == "lost/reborn"]
    out = [f"#### {name}: {len(ev)} switches, {len(dups)} duplicate moments\n",
           "| cause | switches | of which per-camera tracklet continuous |", "|---|---|---|"]
    out += [f"| {k} | {c[k]} | {cont[k]} |" for k in CAUSES]
    if gaps:
        out.append(f"\nlost/reborn: A ended by {dict(ends)}; gap median {np.median(gaps) / 30:.1f} s "
                   f"(max {max(gaps) / 30:.1f} s), end-to-start distance median {np.median(dists):.1f} m")
    if dups:
        same = sum(d["same_cam"] for d in dups)
        dd = [d["dist"] for d in dups if np.isfinite(d["dist"])]
        out.append(f"duplicates: {same} same camera, {len(dups) - same} different cameras; "
                   f"distance between the two tracks median {np.median(dd):.1f} m")
        pairs = collections.Counter(" vs ".join(sorted(d["cams"])) for d in dups if not d["same_cam"])
        out.append("cross-camera duplicate camera pairs: " + ", ".join(f"{k} {n}" for k, n in pairs.most_common(6)))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--name", default="game_2to3")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--events", action="store_true", help="list every switch")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    gt = load_gt(args.name, run)
    variants = [
        ("court tracker + team colour", WTParams(online_relink=True, use_numbers=False)),
        ("+ colour + numbers (current)", WTParams(online_relink=True)),
        ("+ colour + numbers, team required, 10 s relink", WTParams(online_relink=True, num_needs_team=True, id_gap=300)),
    ]
    md = ""
    for name, p in variants:
        r = analyse(track_run(run, p), gt)
        s = summary(name, r)
        print(s, flush=True)
        if args.events:
            for e in sorted(r["events"], key=lambda e: (e["f1"], e["player"])):
                print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
        md += s + "\n"
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
