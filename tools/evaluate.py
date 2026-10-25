"""Evaluate MV3DT runs from their recorded outputs (no ground truth available).

Source: experiments/<run>/kafka.jsonl.gz (one message per camera per frame; each object has its
global ID and that camera's footLocation3D in world metres), plus console.log (FPS) and gpu.csv.

Metrics
  ids, ids_per_min            unique global IDs (lower = less fragmentation, for the same scene)
  multi_cam_ids               IDs reported by >= 2 cameras
  median_life_s               median ID lifetime (first to last frame seen, any camera)
  interior_births_per_min     new IDs whose first fused position is > 1.5 m inside the court lines:
                              players do not appear mid-court, so these are ID switches / fragments
  xcam_med_m, xcam_p95_m      foot-position disagreement between cameras, same ID and frame
  xcam_<pair>                 same, per camera pair (median)
  missed_merge_rate           pairs of *different* IDs from different cameras within 0.5 m in the same
                              frame, per frame (person seen by two cameras but not associated)
  inside_court, inside_margin fraction of foot points inside the court / inside court + 1 m
  pc_life_med_s               per camera: median track lifetime (that camera's view only), then the median
                              over cameras -- isolates single-view tracker fragmentation
  pc_ids_per_min              per camera IDs per minute, summed over cameras
  fps, elapsed_s, gpu_mem_peak_mib, gpu_util_med

Cleaned (tools/stitch.py: court filter on bbox bottom, duplicate merge, stitching, >= 1 s tracks)
  c_kept                      fraction of detections whose bbox bottom is on the court (+0.75 m)
  c_ids, c_ids_per_min        tracks after clean-up
  c_concurrent_med / _p95     tracks present on court in a frame (compare with the ~10 people on court)
  c_interior_births_per_min   cleaned tracks starting > 1.5 m inside the court lines
  c_median_life_s             median cleaned track lifetime
  Writes experiments/<run>/tracks_clean.csv.

World tracker (tools/world_tracker.py: court-plane Kalman tracker over all cameras' detections, MV3DT IDs ignored)
  w_ids_per_min, w_alive_med / _p95 (confirmed tracks alive per frame, incl. coasting -- compare with ~10),
  w_interior_births_per_min, w_median_life_s, w_cams_mean (cameras per update). Writes tracks_world.csv.
  With appearance (tools/appearance.py): w_purity (mean share of a track's team-labelled detections that
  agree with its majority team; tracks with >= 10 labels), w_impure (share of those tracks below 0.8 purity:
  likely cross-team swaps), w_team_known (share of tracks with a confident team).

  python -m tools.evaluate c60_nopose c60_pose ... [--md out.md] [--csv out.csv]
"""

import argparse
import collections
import csv
import gzip
import itertools
import json
import re
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.court import FT, HALF_FLAG_FT, HALF_LOUNGE_FT, WIDTH_FT
from tools.stitch import Params, clean, save_tracks

SENSORS = {"Camera1": "NL", "Camera2": "FL", "Camera3": "NR", "Camera4": "FR"}  # msgconv: sorted video order
CAM_ORDER = {c: i for i, c in enumerate(SENSORS.values())}
X_MIN, X_MAX, Y_HALF = -HALF_FLAG_FT * FT, HALF_LOUNGE_FT * FT, WIDTH_FT * FT / 2
FPS_VIDEO = 30


def inside(xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    return (xy[:, 0] >= X_MIN - margin) & (xy[:, 0] <= X_MAX + margin) & (np.abs(xy[:, 1]) <= Y_HALF + margin)


def load_kafka(path: Path):
    """Return {frame: {sensor: {gid: (x, y)}}}."""
    frames = collections.defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            m = json.loads(line)
            objs = {}
            for o in m.get("objects", []):
                fl = o.get("info", {}).get("footLocation3D")
                if fl:
                    objs[o["id"]] = tuple(float(v) for v in fl.split(","))
            frames[int(m["id"])][SENSORS.get(m["sensorId"], m["sensorId"])] = objs
    return frames


def perf(run: Path) -> dict:
    out = {}
    con = run / "console.log"
    if con.exists():
        vals = []
        for line in con.read_text(errors="ignore").splitlines():
            if re.match(r"\*\*PERF:\s+[\d.]", line):
                v = [float(x) for x in re.findall(r"([\d.]+) \(", line)]
                if v and min(v) > 0:
                    vals.append(np.mean(v))
        if len(vals) > 2:
            out["fps"] = float(np.median(vals[1:-1]))
    rt = run / "run.txt"
    if rt.exists():
        m = re.search(r"pipeline_elapsed_s=(\d+)", rt.read_text())
        if m:
            out["elapsed_s"] = int(m.group(1))
    g = run / "gpu.csv"
    if g.exists():
        rows = list(csv.reader(g.open()))[1:]
        mem = np.array([float(r[1]) for r in rows if len(r) > 3])
        util = np.array([float(r[3]) for r in rows if len(r) > 3])
        if len(mem):
            out["gpu_mem_peak_mib"] = float(mem.max())
            busy = util[util > 20]
            out["gpu_util_med"] = float(np.median(busy)) if len(busy) else 0.0
    return out


def metrics(run: Path) -> dict:
    frames = load_kafka(run / "kafka.jsonl.gz")
    fids = sorted(frames)
    minutes = len(fids) / FPS_VIDEO / 60
    first, last, first_pos = {}, {}, {}
    cams = collections.defaultdict(set)
    pts, pair_d = [], collections.defaultdict(list)
    missed = 0
    for f in fids:
        per_cam = frames[f]
        by_gid = collections.defaultdict(dict)
        for cam, objs in per_cam.items():
            for gid, xy in objs.items():
                by_gid[gid][cam] = np.array(xy)
                cams[gid].add(cam)
                pts.append(xy)
        for gid, d in by_gid.items():
            if gid not in first:
                first[gid] = f
                first_pos[gid] = np.mean(list(d.values()), axis=0)
            last[gid] = f
            for a, b in itertools.combinations(sorted(d, key=CAM_ORDER.get), 2):
                pair_d[(a, b)].append(float(np.linalg.norm(d[a] - d[b])))
        # different IDs from different cameras that sit on top of each other
        items = [(cam, gid, xy) for gid, d in by_gid.items() for cam, xy in d.items()]
        for (c1, g1, p1), (c2, g2, p2) in itertools.combinations(items, 2):
            if c1 != c2 and g1 != g2 and np.linalg.norm(p1 - p2) < 0.5:
                missed += 1
    cam_first, cam_last = collections.defaultdict(dict), collections.defaultdict(dict)
    for f in fids:
        for cam, objs in frames[f].items():
            for gid in objs:
                cam_first[cam].setdefault(gid, f)
                cam_last[cam][gid] = f
    pc_life = [np.median([(cam_last[c][g] - cam_first[c][g] + 1) / FPS_VIDEO for g in cam_first[c]]) for c in cam_first]
    pts = np.array(pts)
    births = np.array(list(first_pos.values()))
    interior = inside(births, -1.5).sum() if len(births) else 0
    life = np.array([(last[g] - first[g] + 1) / FPS_VIDEO for g in first])
    all_d = np.concatenate([v for v in pair_d.values()]) if pair_d else np.array([np.nan])
    out = {
        "run": run.name, "frames": len(fids),
        "ids": len(first), "ids_per_min": len(first) / minutes,
        "multi_cam_ids": sum(len(c) > 1 for c in cams.values()),
        "median_life_s": float(np.median(life)),
        "interior_births_per_min": interior / minutes,
        "xcam_med_m": float(np.median(all_d)), "xcam_p95_m": float(np.percentile(all_d, 95)),
        "missed_merge_rate": missed / len(fids),
        "pc_life_med_s": float(np.median(pc_life)) if pc_life else 0.0,
        "pc_ids_per_min": sum(len(v) for v in cam_first.values()) / minutes,
        "inside_court": float(inside(pts).mean()), "inside_margin": float(inside(pts, 1.0).mean()),
    }
    for (a, b), v in sorted(pair_d.items()):
        out[f"xcam_{a}-{b}"] = float(np.median(v))
        out[f"n_{a}-{b}"] = len(v)
    out.update(perf(run))
    if (run / "camInfo_used").is_dir():
        out.update(clean_metrics(run, minutes))
        from tools.world_tracker import save, track_run
        wt = track_run(run)
        save(wt, run / "tracks_world.csv")
        out.update(world_metrics(wt, run, minutes))
    return out


def clip_tracks(tracks, f0: int, f1: int) -> list:
    """Copies of tracks restricted to frames [f0, f1] (outputs, labels, heights); empty tracks dropped."""
    import copy
    out = []
    for t in tracks:
        c = copy.copy(t)
        c.out = [o for o in t.out if f0 <= o[0] <= f1]
        if not c.out:
            continue
        c.labels = [lf for lf in getattr(t, "labels", []) if f0 <= lf[0] <= f1]
        c.reads = [r for r in getattr(t, "reads", []) if f0 <= r[0] <= f1]
        c.dets = [k for k in getattr(t, "dets", []) if f0 <= k[0] <= f1]
        c.heights = [hf for hf in getattr(t, "heights", []) if f0 <= hf[0] <= f1]
        if hasattr(t, "votes"):
            c.votes = collections.Counter(lab for _, lab in c.labels)
        out.append(c)
    return out


def identity_metrics(tracks, minutes: float, min_player_reads: int = 20, switch_run: int = 3,
                     ref_reads: dict | None = None) -> dict:
    """Jersey numbers as stand-in ground truth (see tools/jersey.py). With ref_reads ((frame, cam, gid) ->
    (number, score)) every variant is measured against the same, complete set of reads via the detections
    each track used; otherwise the reads the tracker itself consumed (t.reads) are used."""
    from tools.world_tracker import WTParams
    p = WTParams()
    per_id = collections.defaultdict(list)
    id_reads = collections.Counter()
    pur_num = pur_den = switches = cov_num = cov_den = 0
    for t in tracks:
        if ref_reads is not None:
            reads = sorted((k[0], *ref_reads[k]) for k in getattr(t, "dets", []) if k in ref_reads)
        else:
            reads = getattr(t, "reads", [])
        nums = [n for _, n, _ in reads]
        team = t.firm_team(p) if hasattr(t, "firm_team") else None
        cov_den += len(t.out)
        if len(nums) >= 3:
            c = collections.Counter(nums)
            pur_num += c.most_common(1)[0][1]
            pur_den += len(nums)
        if nums:
            c = collections.Counter(nums)
            num, k = c.most_common(1)[0]
            if len(nums) >= p.num_min_reads and k / len(nums) >= p.num_share:
                per_id[(team, num)].append(t)
                cov_num += len(t.out)
            for n in nums:
                id_reads[(team, n)] += 1
        cur, run_n, run_len = None, None, 0     # persistent number changes inside the track
        for n in nums:
            run_len = run_len + 1 if n == run_n else 1
            run_n = n
            if run_len == switch_run:
                if cur is not None and n != cur:
                    switches += 1
                cur = n
    players = [k for k, v in id_reads.items() if v >= min_player_reads and k[1] is not None]
    tpp = [len(per_id.get(k, [])) for k in players]
    return {
        "id_players": len(players),
        "id_tracks_per_player": float(np.mean(tpp)) if tpp else float("nan"),
        "id_read_purity": pur_num / pur_den if pur_den else float("nan"),
        "id_switches_per_min": switches / minutes,
        "id_coverage": cov_num / cov_den if cov_den else float("nan"),
    }


def world_metrics(tracks, run: Path, minutes: float | None = None, frames: tuple | None = None,
                  ref_reads: dict | None = None) -> dict:
    if frames is not None:
        tracks = clip_tracks(tracks, *frames)
        minutes = (frames[1] - frames[0] + 1) / FPS_VIDEO / 60
    if minutes is None:
        frames = {f for t in tracks for f, *_ in t.out}
        minutes = (max(frames) - min(frames) + 1) / FPS_VIDEO / 60 if frames else 1.0
    alive, visible = collections.Counter(), collections.Counter()
    births, life, cams = 0, [], []
    for t in tracks:
        f0, f1 = t.out[0][0], t.out[-1][0]
        for f in range(f0, f1 + 1):
            alive[f] += 1
        for f in {o[0] for o in t.out}:
            visible[f] += 1
        life.append((f1 - f0 + 1) / FPS_VIDEO)
        births += bool(inside(np.array([t.out[0][1:3]]), -1.5)[0])
        cams += [n for *_, n in t.out]
    a = np.array([alive.get(f, 0) for f in range(min(alive, default=0), max(alive, default=0) + 1)])
    pur = []
    for t in tracks:
        labs = [lab for _, lab in getattr(t, "labels", [])]
        if len(labs) >= 10:
            pur.append(collections.Counter(labs).most_common(1)[0][1] / len(labs))
    from tools.world_tracker import WTParams
    known = [t.team(WTParams()) is not None for t in tracks] if tracks and hasattr(tracks[0], "team") else []
    out_id = (identity_metrics(tracks, minutes, ref_reads=ref_reads)
              if ref_reads is not None or any(getattr(t, "reads", None) for t in tracks) else {})
    return {
        **out_id,
        "w_purity": float(np.mean(pur)) if pur else float("nan"),
        "w_impure": float(np.mean(np.array(pur) < 0.8)) if pur else float("nan"),
        "w_team_known": float(np.mean(known)) if known else float("nan"),
        "w_ids": len(tracks), "w_ids_per_min": len(tracks) / minutes,
        "w_alive_med": float(np.median(a)) if len(a) else 0.0,
        "w_alive_p95": float(np.percentile(a, 95)) if len(a) else 0.0,
        "w_visible_med": float(np.median([visible.get(f, 0) for f in range(min(alive, default=0), max(alive, default=0) + 1)])) if alive else 0.0,
        "w_interior_births_per_min": births / minutes,
        "w_median_life_s": float(np.median(life)) if life else 0.0,
        "w_cams_mean": float(np.mean(cams)) if cams else 0.0,
    }


def clean_metrics(run: Path, minutes: float, p: Params | None = None) -> dict:
    tracks, steps = clean(run, inside, p)
    save_tracks(tracks, run / "tracks_clean.csv")
    count = collections.Counter()
    births = 0
    for t in tracks.values():
        on = inside(t.xy, 0.75)
        for f in t.frames[on]:
            count[int(f)] += 1
        births += bool(inside(t.xy[:1], -1.5)[0])
    conc = np.array([count.get(f, 0) for f in range(min(count, default=0), max(count, default=0) + 1)])
    life = np.array([(t.end - t.start + 1) / FPS_VIDEO for t in tracks.values()]) if tracks else np.array([0.0])
    return {
        "c_kept": steps["kept_detections"] / max(1, steps["detections"]),
        "c_ids": len(tracks), "c_ids_per_min": len(tracks) / minutes,
        "c_concurrent_med": float(np.median(conc)) if len(conc) else 0.0,
        "c_concurrent_p95": float(np.percentile(conc, 95)) if len(conc) else 0.0,
        "c_interior_births_per_min": births / minutes,
        "c_median_life_s": float(np.median(life)),
        **{f"step_{k}": v for k, v in steps.items()},
    }


COLUMNS = [("run", "{}"), ("fps", "{:.1f}"), ("gpu_mem_peak_mib", "{:.0f}"), ("ids_per_min", "{:.0f}"),
           ("interior_births_per_min", "{:.1f}"), ("median_life_s", "{:.1f}"), ("multi_cam_ids", "{}"),
           ("xcam_med_m", "{:.2f}"), ("xcam_p95_m", "{:.2f}"), ("missed_merge_rate", "{:.2f}"),
           ("inside_court", "{:.1%}"), ("inside_margin", "{:.1%}")]
TRACKER_COLUMNS = [("run", "{}"), ("fps", "{:.1f}"), ("pc_life_med_s", "{:.1f}"), ("pc_ids_per_min", "{:.0f}"),
                   ("ids_per_min", "{:.0f}"), ("interior_births_per_min", "{:.1f}"), ("xcam_p95_m", "{:.2f}"),
                   ("c_ids_per_min", "{:.0f}"), ("c_concurrent_med", "{:.0f}"), ("c_median_life_s", "{:.1f}")]
WORLD_COLUMNS = [("run", "{}"), ("w_ids_per_min", "{:.0f}"), ("w_alive_med", "{:.0f}"), ("w_alive_p95", "{:.0f}"),
                 ("w_visible_med", "{:.0f}"), ("w_interior_births_per_min", "{:.1f}"), ("w_median_life_s", "{:.1f}"), ("w_cams_mean", "{:.2f}"),
                 ("w_purity", "{:.1%}"), ("w_impure", "{:.0%}"), ("w_team_known", "{:.0%}")]
CLEAN_COLUMNS = [("run", "{}"), ("c_kept", "{:.0%}"), ("c_ids_per_min", "{:.0f}"), ("c_concurrent_med", "{:.0f}"),
                 ("c_concurrent_p95", "{:.0f}"), ("c_interior_births_per_min", "{:.1f}"), ("c_median_life_s", "{:.1f}")]


def markdown(rows: list[dict], columns=COLUMNS) -> str:
    head = "| " + " | ".join(c for c, _ in columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(fmt.format(r[c]) if c in r and r[c] is not None else "-" for c, fmt in columns) + " |\n"
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names under experiments/ (or paths)")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    rows = []
    for r in args.runs:
        p = Path(r) if Path(r).is_dir() else ROOT / "experiments" / r
        rows.append(metrics(p))
    table = "Raw MV3DT output\n\n" + markdown(rows) + "\nAfter clean-up (court filter, merge, stitch, >= 1 s)\n\n" \
        + markdown(rows, CLEAN_COLUMNS) + "\nTracker focus\n\n" + markdown(rows, TRACKER_COLUMNS) \
        + "\nWorld tracker (court-plane, MV3DT IDs ignored)\n\n" + markdown(rows, WORLD_COLUMNS)
    print(table)
    if args.md:
        args.md.write_text(table)
    if args.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
