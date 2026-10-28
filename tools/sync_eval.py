"""Identity accuracy of runs on time-aligned videos against runs on the original ones (GitHub issue #3).

For each run: MV3DT's own global IDs and the court tracker's default output (world_tracker.track_default), identity
IDF1 and switches on both labelled minutes, the end-to-end share (correctly identified / all labelled), and the
median distance between two cameras' MV3DT foot points of the same person at the same frame number. Runs on
a dataset cut with per-camera offsets (tools/make_clips --offsets) have each camera's frames shifted; for scoring,
every detection key is mapped back to the original frame numbers, so that one labelled moment stays one moment (the
switch count depends on it; IDF1 doesn't).

  python -m tools.sync_eval mount_r1 mount_r2 mount_r3 sync_r1 sync_r2 sync_r3 [--md out.md]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from tools import ROOT

MINUTES = ("game_2to3", "game_4to5")


@dataclass
class _T:
    dets: list = field(default_factory=list)


def _unshift(tracks, off: dict) -> list:
    return [_T([(int(k[0]) + off.get(str(k[1]), 0), str(k[1]), str(k[2])) for k in t.dets]) for t in tracks]


def score_run(name: str) -> dict:
    from tools.final_eval import mv3dt_tracks
    from tools.gt_eval import frame_offsets, load_gt, score
    from tools.world_tracker import identity_tracks, track_default, track_run
    run = ROOT / "experiments" / name
    off = frame_offsets(run)
    G = identity_tracks(track_default(run), track_run.last_det_ident)
    M = mv3dt_tracks(run)
    from tools.sync_check import pair_distances
    out = {"run": name, "offsets": off,
           "dist": {k: float(np.median(v[0])) for k, v in pair_distances(run).items() if len(v[0]) > 500}}
    for n in MINUTES:
        gt = {(f + off.get(c, 0), c, g): v for (f, c, g), v in load_gt(n, run).items()}
        full = len(load_gt(n))
        for label, T in (("mv3dt", M), ("court", G)):
            s = score(_unshift(T, off), gt)
            out[(label, n)] = (s["idf1"], s["switches"], s["idf1"] * len(gt) / full)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    a = ap.parse_args()
    with ProcessPoolExecutor(min(6, len(a.runs))) as ex:
        res = list(ex.map(score_run, a.runs))
    groups = {}
    for r in res:
        groups.setdefault("clip start offsets (" + ", ".join(f"{k} {v}" for k, v in r["offsets"].items()) + ")"
                          if r["offsets"] else "original clips (no offsets)", []).append(r)
    pairs = [("NL", "FL"), ("FL", "FR"), ("NR", "FR"), ("FL", "NR"), ("NL", "FR")]
    lines = ["| videos | runs | MV3DT IDF1 | MV3DT switches | court tracker IDF1 | court tracker switches | "
             "court tracker end-to-end | " + " | ".join(f"{a}-{b} distance" for a, b in pairs) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(pairs)]
    for g, rs in groups.items():
        def col(label, i):
            return [r[(label, n)][i] for r in rs for n in MINUTES]
        m, c = col("mv3dt", 0), col("court", 0)
        lines.append(f"| {g} | {', '.join(r['run'] for r in rs)} | {np.mean(m):.1%} ({min(m):.0%}-{max(m):.0%}) | "
                     f"{np.mean(col('mv3dt', 1)):.0f} | {np.mean(c):.1%} ({min(c):.0%}-{max(c):.0%}) | "
                     f"{np.mean(col('court', 1)):.0f} | {np.mean(col('court', 2)):.1%} | "
                     + " | ".join(f"{np.mean([r['dist'][p] for r in rs if p in r['dist']]):.2f} m"
                                  if any(p in r['dist'] for r in rs) else "—" for p in pairs) + " |")
    lines += ["", "Per run and minute (court tracker IDF1 / MV3DT IDF1): " + "; ".join(
        f"{r['run']} {n[5:]} {r[('court', n)][0]:.0%} / {r[('mv3dt', n)][0]:.0%}" for r in res for n in MINUTES)]
    md = "\n".join(lines) + "\n"
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
