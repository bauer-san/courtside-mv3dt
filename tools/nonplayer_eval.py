"""Who ends up in the court tracker's output: players, referees, non-players (spectators, bench, staff) and unclear.

Every labelled detection of the two labelled minutes (labels "none" = not a player, "unclear", "ref...", or a player
identity; all court-filtered detections of a moment were labelled) is looked up in the deployable court tracker's
output (confirmed tracks, as sent in the per-frame messages): output at all, output with an identity. For non-players
also where they are (distance to the nearest court line, negative = outside), whether their track ever moves, and the
body height estimated from the box (FL/FR, tools/appearance), as candidate cues for dropping seated people.

  mount runs   Python reference, main causal configuration (world_tracker.track_default: stored bias, online team
               model, no spectator filter)
  ct3 runs     the live DeepStream output, replayed from its recorded inputs (identical to the live output)

  python -m tools.nonplayer_eval mount_r1 mount_r2 mount_r3 ct3_r1 ct3_r2 ct3_r3 [--jobs 6] [--md out.md]
  python -m tools.nonplayer_eval mount_r1 mount_r2 mount_r3 --no-seated   # without the seated-person rule (mount runs)
Also reports identity IDF1 per labelled minute (identity output), so a rule's cost on players is visible.
"""

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools import ROOT

MINUTES = ("game_2to3", "game_4to5")
CATS = ("player", "referee", "not a player", "unclear")


def categories(run: Path) -> dict:
    """(frame, cam, gid) -> (category, label) for every labelled detection, carried over to `run`."""
    from tools.gt_eval import remap
    out = {}
    for name in MINUTES:
        d = ROOT / "gt" / name
        tasks = json.loads((d / "tasks.json").read_text())
        lab_run = {}
        for m in tasks["moments"]:
            for g in m["groups"]:
                p = d / "labels" / "labels" / f"{g['id']}.json"
                if not p.exists():
                    continue
                lab = json.loads(p.read_text())
                body = lab.get("data", lab)
                for c in g["crops"]:
                    i = body.get("crops", {}).get(f"{c['cam']}:{c['gid']}") or body["label"]
                    k = ("not a player" if i == "none" else "unclear" if i == "unclear" else
                         "referee" if i.startswith("ref") else "player")
                    lab_run[(m["frame"], c["cam"], str(c["gid"]))] = (k, i)
        src = ROOT / "experiments" / tasks["run"]
        out.update(remap(lab_run, src, run) if src.resolve() != run.resolve() else lab_run)
    return out


def job(args) -> dict:
    run_name, other, seated_gap = args
    from tools import world_tracker as wtm
    from tools.appearance import load_appearance
    from tools.world_tracker import _edge_dist, load_detections
    run = ROOT / "experiments" / run_name
    if (run / "courttrack_inputs.bin").exists():
        from tools.online_teams import replay_inputs
        tracks, det_ident = replay_inputs(run / "courttrack_inputs.bin")
    else:
        p = wtm.WTParams(online_relink=True) if seated_gap == "default" else wtm.WTParams(online_relink=True, seated_gap=seated_gap)
        tracks = wtm.track_default(run, p)
        det_ident = wtm.track_run.last_det_ident
    from tools.gt_eval import load_gt, score
    from tools.world_tracker import identity_tracks
    G = identity_tracks(tracks, det_ident)
    idf1 = [score(G, load_gt(n, run))["idf1"] for n in MINUTES]
    owner = {}
    for t in tracks:
        for k in t.dets:
            owner[(int(k[0]), str(k[1]), str(k[2]))] = t
    moving = {}
    for t in tracks:
        xy = np.array([o[1:3] for o in t.out])
        c = np.median(xy, axis=0)
        moving[id(t)] = float(np.percentile(np.linalg.norm(xy - c, axis=1), 90))
    d = load_detections(run / "kafka.jsonl.gz")
    pos = {(int(f), str(c), str(g)): (x, y) for f, c, g, x, y in zip(d["frame"], d["cam"], d["gid"], d["x"], d["y"])}
    a = load_appearance(run)
    height = {(int(f), str(c), str(g)): float(h) for f, c, g, h in zip(a["frame"], a["cam"], a["gid"], a["height"])
              if np.isfinite(h)}
    rows = []
    for k, (cat, lab) in categories(run).items():
        t = owner.get(k)
        p = pos.get(k)
        rows.append(dict(cat=cat, label=lab, out=t is not None, ident=det_ident.get(k) or "",
                         edge=_edge_dist(*p) if p else None, spread=moving.get(id(t)) if t else None,
                         height=height.get(k), key=k))
    return {"run": run_name, "rows": rows, "idf1": idf1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--no-seated", action="store_true", help="without the seated-person rule (mount runs only; the live runs "
                    "were filtered in the pipeline)")
    a = ap.parse_args()
    mount = [r for r in a.runs if r.startswith("mount_")]
    others = {r: mount[(i + 1) % len(mount)] if mount else None for i, r in enumerate(mount)}
    with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:
        res = list(ex.map(job, [(r, others.get(r), None if a.no_seated else "default") for r in a.runs]))
    lines = []
    for group, names in (("Python reference, main configuration (mount runs)", [r for r in a.runs if r in mount]),
                         ("Live in DeepStream", [r for r in a.runs if r not in mount])):
        rows = [x for r in res if r["run"] in names for x in r["rows"]]
        if not rows:
            continue
        idf = [x for r in res if r["run"] in names for x in r["idf1"]]
        rule = " (without the seated-person rule)" if a.no_seated and group.startswith("Python") else ""
        lines += [f"### {group}{rule}: {', '.join(names)}", "",
                  f"Identity IDF1 (players): {np.mean(idf):.1%} ({min(idf):.0%}-{max(idf):.0%}); per run-minute "
                  + ", ".join(f"{x:.1%}" for x in idf), "",
                  "| labelled as | detections | in the output | with an identity |", "|---|---|---|---|"]
        for c in CATS:
            xs = [x for x in rows if x["cat"] == c]
            if not xs:
                continue
            o = sum(x["out"] for x in xs)
            i = sum(bool(x["ident"]) for x in xs)
            lines.append(f"| {c} | {len(xs)} | {o} ({o / len(xs):.0%}) | {i} ({i / len(xs):.0%}) |")
        npl = [x for x in rows if x["cat"] == "not a player"]
        if npl:
            e = np.array([x["edge"] for x in npl if x["edge"] is not None])
            lines += ["", f"Not-a-player detections ({len(npl)}):",
                      f"- position: {np.mean(e < 0):.0%} outside the lines, {np.mean((e >= 0) & (e < 1.5)):.0%} within "
                      f"1.5 m inside, {np.mean(e >= 1.5):.0%} further in (median {np.median(e):+.2f} m from the nearest line)"]
            outn = [x for x in npl if x["out"]]
            if outn:
                sp = np.array([x["spread"] for x in outn])
                lines.append(f"- in the output: {len(outn)}; their tracks' 90th-percentile spread around the median "
                             f"position: median {np.median(sp):.2f} m, {np.mean(sp < 1.0):.0%} under 1 m")
                ids = collections.Counter(x["ident"] for x in outn if x["ident"])
                if ids:
                    lines.append("- identities given to them: " + ", ".join(f"{k} x{v}" for k, v in ids.most_common(8)))
            hs = [x["height"] for x in npl if x["height"] is not None]
            hp = [x["height"] for x in rows if x["cat"] == "player" and x["height"] is not None]
            if hs and hp:
                lines.append(f"- body height from the box (FL/FR crops): non-players median {np.median(hs):.2f} m "
                             f"(n={len(hs)}, {np.mean(np.array(hs) < 1.4):.0%} under 1.4 m); players median "
                             f"{np.median(hp):.2f} m (n={len(hp)}, {np.mean(np.array(hp) < 1.4):.0%} under 1.4 m)")
        lines.append("")
    md = "\n".join(lines)
    print(md)
    if a.md:
        a.md.write_text(md)


if __name__ == "__main__":
    main()
