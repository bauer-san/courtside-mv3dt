"""End-to-end identity accuracy of the deployment variants (INT8, detector interval, no display; report section 6.1).

End-to-end = correctly identified labelled detections / all labelled detections, so people a variant fails to detect
count as errors (identity IDF1 alone is computed on the labels the variant detected, carried over by box overlap).
Court tracker in its default configuration, as in the report's main results.

  python -m tools.deploy_eval [--jobs 8] [--md reports/deploy_end_to_end.md]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools import ROOT

GROUPS = {"FP16, every frame (with display)": ["mount_r1", "mount_r2", "mount_r3"],
          "FP16, every frame, no display": ["nodisp_r1"],
          "INT8, every frame": ["int8_r1", "int8_r2", "int8_r3"],
          "INT8, every frame, detector threshold 0.1": ["int8t10_r1", "int8t10_r2", "int8t10_r3"],
          "FP16, every 2nd frame": ["int1_r1", "int1_r2", "int1_r3"],
          "FP16, every 3rd frame": ["int2_r1", "int2_r2", "int2_r3"],
          "INT8, every 2nd frame": ["int8int1_r1", "int8int1_r2", "int8int1_r3"]}
MINUTES = ("game_2to3", "game_4to5")


def score_run(r: str) -> list[tuple[float, float]]:
    """(identity IDF1 on detected labels, share of labels detected) per labelled minute."""
    from tools.gt_eval import load_gt, score
    from tools.world_tracker import WTParams, identity_tracks, track_run
    run = ROOT / "experiments" / r
    G = identity_tracks(track_run(run, WTParams(online_relink=True)), track_run.last_det_ident)
    out = []
    for n in MINUTES:
        gt = load_gt(n, run)
        out.append((score(G, gt)["idf1"], len(gt) / len(load_gt(n))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    runs = [r for rs in GROUPS.values() for r in rs]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(zip(runs, ex.map(score_run, runs)))
    lines = ["| variant | runs | identity IDF1 on detected labels | labelled people detected | "
             "end-to-end: correctly identified / all labelled |", "|---|---|---|---|---|"]
    for name, rs in GROUPS.items():
        xs = [x for r in rs for x in res[r]]
        idf = [i for i, _ in xs]
        det = [d for _, d in xs]
        e2e = [i * d for i, d in xs]
        lines.append(f"| {name} | {len(rs)} | {np.mean(idf):.1%} ({min(idf):.0%}-{max(idf):.0%}) | {np.mean(det):.1%} | "
                     f"**{np.mean(e2e):.1%}** ({min(e2e):.0%}-{max(e2e):.0%}) |")
    md = "\n".join(lines) + "\n"
    print(md)
    if a.md:
        a.md.write_text(md)


if __name__ == "__main__":
    main()
