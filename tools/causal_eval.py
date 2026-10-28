"""The court tracker's causal inputs and their offline counterparts (report section 4.6).

The main configuration (world_tracker.track_default) is causal, as deployed: camera bias fixed at installation (stored
in camInfo, measured on other runs), team colours from the automatic online model (3 clusters), no spectator filter.
Each input is replaced with an alternative, including the offline inputs prepared from the whole recording (an upper
reference a live system can't have), and scored paired against the main configuration on every run x labelled minute.

  bias_whole    camera bias estimated over the whole run (offline)
  bias_2min     camera bias estimated live from the first 2 minutes
  bias_none     no camera bias correction
  online4       online team model with 4 clusters (two teams, referees, other)
  team_offline  offline team model: clustered over the game, clusters named by hand
  team_2min     offline team model from the first 2 minutes only, clusters named after the nearest hand-named
                cluster (standing in for an operator)
  spectator     offline spectator filter (whole tracks) and post-stitching
  offline_all   all offline inputs: whole-run bias, hand-named team model, spectator filter (the former default)
The main configuration also runs with seeds 0-4 of the online model's first clustering; all rows use seed 0.

  python -m tools.causal_eval s_int8t10_r1 s_int8t10_r2 s_int8t10_r3 [--jobs 8] [--md out.md]
"""

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PosixPath

import numpy as np

from tools import GAME_END, ROOT

MINUTES = ("game_2to3", "game_4to5")
SEEDS = range(5)
ROWS = [("main", "main (causal): fixed bias from installation, online 3-cluster team model, no spectator filter ¹"),
        ("bias_whole", "camera bias: estimated over the whole run (offline)"),
        ("bias_2min", "camera bias: estimated live from the first 2 minutes"),
        ("bias_none", "camera bias: none"),
        ("online4", "team colours: online model, 4 clusters (two teams, referees, other)"),
        ("team_offline", "team colours: offline model, clustered over the game, clusters named by hand"),
        ("team_2min", "team colours: offline model from the first 2 minutes only ²"),
        ("spectator", "spectator filter and post-stitching (offline, whole tracks)"),
        ("offline_all", "all offline: whole-run bias, hand-named team model, spectator filter (upper reference)")]
OFFLINE_TEAMS = ("team_offline", "team_2min", "offline_all")


def _bias_of(run: Path, max_frame: int | None = None) -> dict:
    from tools.evaluate import inside
    from tools.stitch import Params as StitchParams
    from tools import world_tracker as wtm
    d = wtm.load_detections(run / "kafka.jsonl.gz")
    keep = wtm.ground_filter(d, wtm.load_cameras(run / "camInfo_used"), inside, StitchParams(margin=0.3))
    bf = wtm._by_frame(d, keep)[0]
    if max_frame is not None:
        bf = {f: v for f, v in bf.items() if f < max_frame}
    return wtm.estimate_bias(bf)


def _team_2min(run: Path, out: Path, upto: int = 3600) -> None:
    """Team model from the run's first `upto` frames (as tools.appearance.cluster_teams, k = 5), clusters named
    after the nearest cluster of the run's hand-named model."""
    import cv2
    import yaml
    from tools import appearance as ap
    a = ap.load_appearance(run)
    sel = ap.good_crops(a) & (a["frame"] < upto)
    a = {k: v[sel] for k, v in a.items()}
    X0 = np.sqrt(a["hist"].astype(np.float32))
    cam_mean = {c: X0[a["cam"] == c].mean(0) for c in np.unique(a["cam"])}
    cam_mean["_global"] = X0.mean(0)
    X = ap.features(a, cam_mean)
    cv2.setRNGSeed(0)
    _, lab, cent = cv2.kmeans(X, 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-5), 10,
                              cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    d = np.sqrt(((X[:, None, :] - cent[None]) ** 2).sum(-1))[np.arange(len(X)), lab]
    radius = np.array([np.percentile(d[lab == c], ap.LABEL_DIST_Q) for c in range(5)])
    ref = np.load(run / "team_model.npz")
    ref_map = yaml.safe_load((run / "teams.yml").read_text())["mapping"]
    mapping = {i: ref_map[int(np.argmin(((ref["centres"] - c) ** 2).sum(1)))] for i, c in enumerate(cent)}
    np.savez(out / "team_model.npz", centres=cent, radius=radius, cams=np.array([c for c in cam_mean if c != "_global"]),
             cam_means=np.array([cam_mean[c] for c in cam_mean if c != "_global"]), global_mean=cam_mean["_global"])
    (out / "teams.yml").write_text(yaml.dump({"max_frame": GAME_END, "mapping": mapping}))


def job(args) -> tuple:
    run_name, variant, seed = args
    from tools import appearance as ap
    from tools import world_tracker as wtm
    from tools.gt_eval import load_gt, score
    from tools.online_teams import OTParams
    from tools.world_tracker import WTParams, identity_tracks, track_default, track_run
    run = ROOT / "experiments" / run_name
    kw = {"correct_bias": "stored", "post": False}
    if variant == "bias_whole":
        kw["correct_bias"] = True
    elif variant == "bias_2min":
        b = _bias_of(run, 3600)
        wtm.estimate_bias = lambda bf, **k: b
        kw["correct_bias"] = True
    elif variant == "bias_none":
        kw["correct_bias"] = False
    elif variant == "spectator":
        kw["post"] = True
    elif variant == "offline_all":
        kw = {"correct_bias": True, "post": True}
    op = OTParams(k=4, ref_label="ref", seed=seed) if variant == "online4" else OTParams(seed=seed)
    tmp = None
    if variant == "team_2min":
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name)
        _team_2min(run, out)

        class Redirect(PosixPath):
            def __truediv__(self, name):
                return out / name if name in ("team_model.npz", "teams.yml") else PosixPath(self) / name
        o_tp, o_cf = ap.team_probs, wtm.colour_features
        ap.team_probs = lambda r, **k: o_tp(Redirect(r), **k)
        wtm.colour_features = lambda r: o_cf(Redirect(r))
    if variant in OFFLINE_TEAMS:
        T = track_run(run, WTParams(online_relink=True), **kw)
    else:
        T = track_default(run, op=op, **kw)
    G = identity_tracks(T, track_run.last_det_ident)
    if tmp is not None:
        tmp.cleanup()
    return run_name, variant, seed, [score(G, load_gt(n, run))["idf1"] for n in MINUTES]


def _bias_source() -> str:
    import re
    from tools.calib_bias import CAMINFO
    m = re.search(r"Measured with tools/calib_bias.py on (.*?)\.\n", next(CAMINFO.glob("*.yml")).read_text())
    return m.group(1) if m else "?"


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("runs", nargs="+")
    ap_.add_argument("--jobs", type=int, default=1)
    ap_.add_argument("--md", type=Path)
    a = ap_.parse_args()
    jobs = [(r, v, sd) for r in a.runs for v, _ in ROWS for sd in (SEEDS if v == "main" else [0])]
    with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:   # jobs patch module functions: fresh process each
        res = list(ex.map(job, jobs))
    by = {}
    for r, v, sd, xs in res:
        by.setdefault((v, sd), {})[r] = xs
    base = by[("main", 0)]
    lines = ["| Configuration | Identity IDF1 | Paired difference | Pairs better / worse |", "|---|---|---|---|"]
    for v, label in ROWS:
        xs = [x for r in a.runs for x in by[(v, 0)][r]]
        d = np.array([x - b for r in a.runs for x, b in zip(by[(v, 0)][r], base[r])])
        if v == "main":
            lines.append(f"| **{label}** | **{np.mean(xs):.1%}** ({min(xs):.0%}-{max(xs):.0%}) | — | — |")
            seed_means = [np.mean([x for r in a.runs for x in by[("main", sd)][r]]) for sd in SEEDS]
        else:
            lines.append(f"| {label} | {np.mean(xs):.1%} ({min(xs):.0%}-{max(xs):.0%}) | {100 * d.mean():+.1f} | "
                         f"{int((d > 0.005).sum())} / {int((d < -0.005).sum())} |")
    foot = ["",
            f"¹ Seed 0 of the online model's first clustering (all rows); seeds 0-{len(SEEDS) - 1}: "
            + ", ".join(f"{m:.1%}" for m in seed_means) + f" (mean {np.mean(seed_means):.1%}).",
            "² Clusters named after the nearest hand-named cluster, standing in for an operator.",
            f"Runs: {', '.join(a.runs)}; fixed camera bias measured on {_bias_source()} (not evaluated here)."]
    md = "\n".join(lines + foot) + "\n"
    print(md)
    if a.md:
        a.md.write_text(md)


if __name__ == "__main__":
    main()
