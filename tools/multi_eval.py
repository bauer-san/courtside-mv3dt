"""Evaluate tracker configurations over several DeepStream runs and both labelled minutes.

DeepStream runs are not repeatable (MV3DT ids, tracklets and jersey reads differ from run to run while the
detections are ~99 % the same), and identity IDF1 moves by several points between runs of the same configuration.
So every configuration is scored on each run x labelled minute; reported: mean, spread, and the difference to the
default paired by run and minute (with how many of the pairs favour the configuration). Every configuration is the
main, causal one (world_tracker.track_default: online team model, stored camera bias) with one change.

  python -m tools.multi_eval s_int8t10_r1 s_int8t10_r2 s_int8t10_r3 [--md out.md]
"""

import argparse
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.final_eval import mv3dt_tracks
from tools.gt_eval import load_gt, score
from tools.world_tracker import WTParams, identity_tracks, track_default, track_run

GT = ("game_2to3", "game_4to5")

# name -> (WTParams overrides, track_run overrides, identity output)
CONFIGS = {
    "default (identity output)": ({}, {}, True),
    # contribution of each part (main table)
    "previous default (no identity freshness)": (dict(id_recent=False, split_majority=False), {}, True),
    "track IDs instead of identities": ({}, {}, False),
    "no per-player look": (dict(player_appearance=False), {}, True),
    "old association (isotropic, plain Mahalanobis)": (dict(aniso=False, cost_logdet=False), {}, True),
    "numbers: old rule (team optional, 60 s)": (dict(num_needs_team=False, id_gap=1800), {}, True),
    "no team colour": ({}, dict(team_labels=False), True),
    "no jersey numbers (track IDs)": (dict(use_numbers=False), {}, False),
    # tried and not adopted (appendix)
    "close-encounter swap resolution": (dict(encounters=True), {}, True),
    "unknown identity after close encounters": (dict(id_doubt=True), {}, True),
    "majority swap split without recent-read identity": (dict(id_recent=False), {}, True),
    "recent-read identity without majority split": (dict(split_majority=False), {}, True),
    "stricter identity rule (3 reads, 70 %)": (dict(num_min_reads=3, num_share=0.7), {}, True),
    "court edges: detections to 0.75 m + 5 s coasting": (dict(edge_margin=0.75, edge_coast=150), {}, True),
    "larger cross-camera noise (0.65 / 0.45 m)": (dict(sigma_radial=0.65, sigma_tangential=0.45), {}, True),
}


def evaluate(run: Path, cfg: tuple, gts: dict) -> dict:
    wp, tk, ident = cfg
    T = track_default(run, WTParams(online_relink=True, **wp), **tk)
    G = identity_tracks(T, track_run.last_det_ident) if ident else T
    return {n: score(G, gt) for n, gt in gts.items()}


def _job(args) -> tuple:
    """One (run, configuration) evaluation (worker process); name None = MV3DT's own global IDs."""
    run, name = args
    gts = {n: load_gt(n, run) for n in GT}
    if name is None:
        T = mv3dt_tracks(run)
        return run, None, [(run.name, n, score(T, gt)["idf1"], score(T, gt)["switches"]) for n, gt in gts.items()]
    return run, name, [(run.name, n, m["idf1"], m["switches"]) for n, m in evaluate(run, CONFIGS[name], gts).items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--configs", nargs="*", help="subset of configuration names")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    args = ap.parse_args()
    runs = [Path(r) if Path(r).is_dir() else ROOT / "experiments" / r for r in args.runs]
    configs = {k: v for k, v in CONFIGS.items() if not args.configs or k in args.configs or k.startswith("default")}
    jobs = [(run, name) for run in runs for name in [None] + list(configs)]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as ex:
            done = list(ex.map(_job, jobs))
    else:
        done = [_job(j) for j in jobs]
    res = {}                                   # config -> [(run, minute, idf1, switches)], in run order
    mv = []
    for run, name, rows in done:
        if name is None:
            mv += rows
        else:
            res.setdefault(name, []).extend(rows)
            print(f"{run.name} {name}: " + "  ".join(f"{x[1]} {x[2]:.0%}" for x in rows), flush=True)
    res = {name: res[name] for name in configs}
    base = {(r, n): v for r, n, v, _ in res["default (identity output)"]}
    lines = [f"### Identity IDF1 over {len(runs)} DeepStream runs x {len(GT)} labelled minutes",
             "", "| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |",
             "|---|---|---|---|---|---|"]
    rows = [("MV3DT global IDs", mv)] + list(res.items())
    for name, xs in rows:
        v = np.array([x[2] for x in xs])
        d = np.array([x[2] - base[(x[0], x[1])] for x in xs])
        sw = np.mean([x[3] for x in xs])
        vs = "—" if name.startswith("default") else f"{d.mean():+.1%}"
        bw = "—" if name.startswith("default") else f"{int((d > 0.005).sum())} / {int((d < -0.005).sum())}"
        lines.append(f"| {name} | {v.mean():.1%} | {v.min():.0%}-{v.max():.0%} | {vs} | {bw} | {sw:.0f} |")
    per = ["", "Per run and minute (default): " + ", ".join(f"{r} {n} {v:.0%}" for (r, n), v in base.items())]
    md = "\n".join(lines + per) + "\n"
    print(md)
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
