"""Court tracker identity logic (GitHub issue #11): score WTParams variants paired per run and labelled minute.

Each variant is the main configuration (world_tracker.track_default) with some WTParams overridden. For every run and
minute it records identity IDF1, the share of labelled detections detected, and the loss breakdown
(tools/loss_breakdown.classify); results are cached in experiments/id_explore/<minutes>/<variant>/<run>.json.

  python -m tools.id_explore base 'swap={"split_share": 0.5}' [--runs js_r1 ...] [--minutes game_2to3,game_4to5]
  python -m tools.id_explore base ... --minutes game_5to6     # held-out check (never tune on it)
"""

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools import ROOT

RUNS = ["js_r1", "js_r2", "js_r3", "dflt_r1", "ctj_r1", "ctj_r2", "ctj_r3"]
OUT = ROOT / "experiments" / "id_explore"


def parse(spec: str) -> tuple[str, dict]:
    name, _, ov = spec.partition("=")
    return name, (json.loads(ov) if ov else {})


def job(args: tuple) -> dict:
    name, ov, run_name, minutes = args
    from tools.gt_eval import load_gt, score
    from tools.loss_breakdown import classify
    from tools.world_tracker import WTParams, identity_tracks, track_default, track_run
    run = ROOT / "experiments" / run_name
    T = track_default(run, WTParams(online_relink=True, **ov))
    di = track_run.last_det_ident
    G = identity_tracks(T, di)
    gts = {n: load_gt(n, run) for n in minutes}
    cls = classify(run, T, di, gts)
    res = {}
    for n, gt in gts.items():
        s = score(G, gt)
        cnt, k = cls[n]
        res[n] = dict(idf1=s["idf1"], switches=s["switches"], det=len(gt) / len(load_gt(n)),
                      loss={c: v / k for c, v in cnt.items() if c != "correct"})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", help="name or name=JSON overrides; the first is the reference")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--minutes", default="game_2to3,game_4to5")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="recompute cached results")
    a = ap.parse_args()
    minutes = a.minutes.split(",")
    specs = [parse(v) for v in a.variants]
    todo, res = [], {}
    for name, ov in specs:
        d = OUT / a.minutes.replace(",", "+") / name
        d.mkdir(parents=True, exist_ok=True)
        for r in a.runs:
            f = d / f"{r}.json"
            if f.exists() and not a.force and json.loads(f.read_text())["overrides"] == ov:
                res[name, r] = json.loads(f.read_text())["result"]
            else:
                todo.append((name, ov, r, minutes))
    with ProcessPoolExecutor(a.jobs) as ex:
        for args, out in zip(todo, ex.map(job, todo)):
            name, ov, r, _ = args
            (OUT / a.minutes.replace(",", "+") / name / f"{r}.json").write_text(
                json.dumps(dict(overrides=ov, result=out), indent=1))
            res[name, r] = out
    ref = specs[0][0]
    cats = sorted({c for v in res.values() for m in v.values() for c in m["loss"]})
    print(f"{len(a.runs)} runs x {len(minutes)} minutes; paired against '{ref}'")
    print(f"{'variant':24s} {'IDF1':>6s} {'e2e':>6s} {'dIDF1':>6s} {'w/l':>5s} {'sw':>5s}  " +
          "  ".join(c.split(":")[0][:6] + ("/" + c.split(",")[-1].strip()[:5] if "," in c else "") for c in cats))
    for name, _ in specs:
        rows = [(res[name, r][n], res[ref, r][n]) for r in a.runs for n in minutes]
        idf = np.array([x["idf1"] for x, _ in rows])
        e2e = np.array([x["idf1"] * x["det"] for x, _ in rows])
        d = idf - np.array([y["idf1"] for _, y in rows])
        loss = collections.defaultdict(list)
        for x, _ in rows:
            for c in cats:
                loss[c].append(x["loss"].get(c, 0.0))
        print(f"{name:24s} {idf.mean():6.1%} {e2e.mean():6.1%} {d.mean() * 100:+6.2f} "
              f"{(d > 0.002).sum():2d}/{(d < -0.002).sum():<2d} {np.mean([x['switches'] for x, _ in rows]):5.1f}  " +
              "  ".join(f"{np.mean(loss[c]):6.1%}" for c in cats))


if __name__ == "__main__":
    main()
