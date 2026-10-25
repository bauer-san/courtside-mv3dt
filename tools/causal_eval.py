"""Offline inputs of the court tracker and their causal replacements (report section 4.6).

The reported default uses three inputs prepared offline from the whole recording: the per-camera position bias
(estimated over the whole run), the team colour model (clustered over the game, clusters named by hand) and the
spectator filter (whole tracks). Each is replaced with what a live system has, and all three together (the
deployable form, run with several seeds of the online team model's clustering), and scored paired against the
default on every run x labelled minute.

  bias_other    fixed bias measured on another run (as at installation)
  bias_2min     bias estimated from the first 2 minutes only
  bias_none     no bias correction
  team_2min     team colours clustered from the first 2 minutes only, clusters named after the nearest hand-named
                cluster (standing in for an operator)
  online3       automatic online team model, 3 clusters (two teams, everyone else)
  online4       automatic online team model, 4 clusters (two teams, referees, other)
  no_spectator  no spectator filter
  deployable    bias_other + online3 + no_spectator, seeds 0-4

  python -m tools.causal_eval mount_r1 mount_r2 mount_r3 [--jobs 8] [--md out.md]
"""

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PosixPath

import numpy as np

from tools import ROOT

MINUTES = ("game_2to3", "game_4to5")
SEEDS = range(5)
ROWS = [("default", "default: whole-run bias, hand-named team model, spectator filter"),
        ("bias_other", "camera bias: fixed, measured on another run (as at installation)"),
        ("bias_2min", "camera bias: estimated live from the first 2 minutes"),
        ("bias_none", "camera bias: none"),
        ("team_2min", "team colours: clustered from the first 2 minutes only ¹"),
        ("online3", "team colours: automatic online model, 3 clusters (two teams, everyone else)"),
        ("online4", "team colours: automatic online model, 4 clusters (two teams, referees, other)"),
        ("no_spectator", "spectator filter: none"),
        ("deployable", "all three: fixed bias + online 3-cluster model + no spectator filter ²")]


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
    (out / "teams.yml").write_text(yaml.dump({"max_frame": 12600, "mapping": mapping}))


def job(args) -> tuple:
    run_name, other_name, variant, seed = args
    from tools import appearance as ap
    from tools import world_tracker as wtm
    from tools.gt_eval import load_gt, score
    from tools.online_teams import OTParams, track_run_online
    from tools.world_tracker import WTParams, identity_tracks, track_run
    run, other = ROOT / "experiments" / run_name, ROOT / "experiments" / other_name
    kw, online = {}, None
    if variant in ("bias_other", "deployable"):
        b = _bias_of(other)
        wtm.estimate_bias = lambda bf, **k: b
    elif variant == "bias_2min":
        b = _bias_of(run, 3600)
        wtm.estimate_bias = lambda bf, **k: b
    elif variant == "bias_none":
        kw["correct_bias"] = False
    if variant in ("no_spectator", "deployable"):
        kw["post"] = False
    if variant in ("online3", "deployable"):
        online = OTParams(k=3, ref_label="other", seed=seed)
    elif variant == "online4":
        online = OTParams(k=4, ref_label="ref", seed=seed)
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
    if online is not None:
        T = track_run_online(run, op=online, **kw)
    else:
        T = track_run(run, WTParams(online_relink=True), **kw)
    G = identity_tracks(T, track_run.last_det_ident)
    if tmp is not None:
        tmp.cleanup()
    return run_name, variant, seed, [score(G, load_gt(n, run))["idf1"] for n in MINUTES]


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("runs", nargs="+")
    ap_.add_argument("--jobs", type=int, default=1)
    ap_.add_argument("--md", type=Path)
    a = ap_.parse_args()
    others = {r: a.runs[(i + 1) % len(a.runs)] for i, r in enumerate(a.runs)}
    jobs = [(r, others[r], v, s) for r in a.runs for v, _ in ROWS for s in (SEEDS if v == "deployable" else [0])]
    with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:   # jobs patch module functions: fresh process each
        res = list(ex.map(job, jobs))
    by = {}
    for r, v, s, xs in res:
        by.setdefault((v, s), {})[r] = xs
    base = by[("default", 0)]
    lines = ["| Input replaced | Identity IDF1 | Paired difference | Pairs better / worse |", "|---|---|---|---|"]
    for v, label in ROWS:
        seeds = SEEDS if v == "deployable" else [0]
        per_seed = []
        diffs = []
        for s in seeds:
            xs = [x for r in a.runs for x in by[(v, s)][r]]
            per_seed.append(np.mean(xs))
            diffs.append([x - b for r in a.runs for x, b in zip(by[(v, s)][r], base[r])])
        xs = [x for s in seeds for r in a.runs for x in by[(v, s)][r]]
        d = np.mean(diffs, axis=0)       # per run-minute, averaged over seeds
        if v == "default":
            lines.append(f"| {label} | {np.mean(xs):.1%} ({min(xs):.0%}-{max(xs):.0%}) | — | — |")
        else:
            lines.append(f"| {label} | {np.mean(xs):.1%} ({min(xs):.0%}-{max(xs):.0%}) | {100 * d.mean():+.1f} | "
                         f"{int((d > 0.005).sum())} / {int((d < -0.005).sum())} |")
        if v == "deployable":
            seed_means = ", ".join(f"{m:.1%}" for m in per_seed)
    foot = [f"", f"¹ Clusters named after the nearest hand-named cluster, standing in for an operator.",
            f"² Mean over {len(SEEDS)} seeds of the online model's first clustering (per-seed means {seed_means}); "
            f"paired differences averaged over the seeds. Other rows: seed 0.",
            f"Runs: {', '.join(a.runs)}; fixed bias of each run measured on: "
            + ", ".join(f"{r} <- {others[r]}" for r in a.runs) + "."]
    md = "\n".join(lines + foot) + "\n"
    print(md)
    if a.md:
        a.md.write_text(md)


if __name__ == "__main__":
    main()
