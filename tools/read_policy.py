"""Jersey reads on every frame, and read policies simulated in the court tracker (GitHub issue #8).

cache   replay DeepStream's jersey reader (tools/hires_jersey: same torso rule, text detector, recogniser and
        bilinear scaling, 1080p clip) on every frame a player's torso is eligible, instead of every 7th
        -> experiments/<run>/jersey_every.npz (frame, cam, gid, number, score; one entry per torso with a text box,
        number '' when nothing was read, as DeepStream's jersey.npz), and jersey_ds7.npz: the entries DeepStream's
        schedule would have read (every 7th frame per MV3DT object), to check the cache against DeepStream
score   court tracker identity (Python reference, default configuration) per policy:
          ds        DeepStream's own reads (jersey.npz)
          ds7       the cache on DeepStream's schedule
          every     every eligible frame (upper bound)
          budget=B  B reading attempts per second over all players: tracks without a number first, identified
                    tracks re-checked every 3 s (WTParams.read_budget, reconfirm_s)
          risk=D    as budget, plus tracks within D m of another track re-read without throttling for 2 s
                    (WTParams.risk_dist; with budget=B: risk=D@B)
        reported: identity IDF1, end-to-end, reading attempts per second (court tracker's count)

  python -m tools.read_policy cache js_r1 js_r2 js_r3
  python -m tools.read_policy score js_r1 js_r2 js_r3 --policies ds ds7 every budget=40 risk=1.0@40 [--md out.md]
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.world_tracker import FPS

MIN_H = 150


def _camera(args) -> tuple:
    """One camera: decode the clip once and read every scheduled torso."""
    from tools.hires_jersey import _sessions, read_torsos
    pos, clip, end, sched = args
    sess = _sessions()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", str(end + 1), "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    rows = []
    for f in range(end + 1):
        raw = p.stdout.read(1920 * 1080 * 3)
        if len(raw) < 1920 * 1080 * 3:
            break
        jobs = sched.get(f)
        if not jobs:
            continue
        im = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)
        for (g, _), r in zip(jobs, read_torsos(sess, [(im, 1, None, box) for _, box in jobs], cv2.INTER_LINEAR)):
            if r is not None:
                rows.append((f, pos, g, r[0], r[1]))
    p.kill()
    return rows


def _save(path: Path, rows: list) -> None:
    rows = sorted(rows)
    np.savez_compressed(path, frame=np.array([r[0] for r in rows], int), cam=np.array([r[1] for r in rows]),
                        gid=np.array([r[2] for r in rows]), number=np.array([r[3] for r in rows]),
                        score=np.array([r[4] for r in rows], np.float32))


def cache(run_name: str) -> None:
    from tools import hires_jersey as hj
    from tools.appearance import run_dataset
    from tools.gt_eval import game_end
    run = ROOT / "experiments" / run_name
    end, ds, boxes = game_end(run), run_dataset(run), hj.load_boxes(run)
    every, ds7 = {}, set()
    for _, pos, _ in CAMERAS:
        hj.REINFER = 1
        every[pos] = hj.schedule(boxes.get(pos, {}), MIN_H, end)
        hj.REINFER = 7
        ds7 |= {(f, pos, g) for f, v in hj.schedule(boxes.get(pos, {}), MIN_H, end).items() for g, _ in v}
    jobs = [(pos, ds / "videos" / f"{cid}_{pos}.mp4", end, every[pos]) for cid, pos, _ in CAMERAS]
    with ProcessPoolExecutor(4) as ex:
        rows = [r for rs in ex.map(_camera, jobs) for r in rs]
    _save(run / "jersey_every.npz", rows)
    _save(run / "jersey_ds7.npz", [r for r in rows if (r[0], r[1], r[2]) in ds7])
    n_try = sum(len(v) for s in every.values() for v in s.values())
    print(f"{run_name}: {n_try} eligible torsos, {len(rows)} with text, {sum(1 for r in rows if r[3])} reads; "
          f"on DeepStream's schedule {len(ds7)} eligible")


def _policy(name: str) -> tuple:
    """policy -> (reads file, WTParams changes)."""
    if name in ("ds", "ds7", "every"):
        return {"ds": "jersey.npz", "ds7": "jersey_ds7.npz", "every": "jersey_every.npz"}[name], {}
    kind, val = name.split("=")
    if kind == "budget":
        return "jersey_every.npz", {"read_budget": float(val)}
    if kind == "risk":
        d, _, b = val.partition("@")
        return "jersey_every.npz", {"risk_dist": float(d), "read_budget": float(b) if b else None}
    raise SystemExit(f"unknown policy {name}")


def _score(args) -> tuple:
    run_name, name = args
    from tools.gt_eval import load_gt, score
    from tools.world_tracker import WTParams, identity_tracks, track_default, track_run
    run = ROOT / "experiments" / run_name
    reads_file, kw = _policy(name)
    T = track_default(run, WTParams(online_relink=True, **kw), reads_file=reads_file)
    G = identity_tracks(T, track_run.last_det_ident)
    frames = max(int(k[0]) for t in T for k in t.dets) + 1
    out = []
    for n in ("game_2to3", "game_4to5"):
        gt = load_gt(n, run)
        out.append((score(G, gt)["idf1"], len(gt) / len(load_gt(n))))
    return run_name, name, out, track_run.last_attempts / (frames / FPS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cache", "score"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--policies", nargs="+", default=["ds", "ds7", "every"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "cache":
        for r in a.runs:
            cache(r)
        return
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_score, [(r, p) for p in a.policies for r in a.runs]))
    lines = [f"### Read policies on {', '.join(a.runs)} (court tracker, {len(a.runs)} runs x 2 labelled minutes)", "",
             "| policy | identity IDF1 | end-to-end | per run and minute | reading attempts per second |",
             "|---|---|---|---|---|"]
    for p in a.policies:
        rs = [x for x in res if x[1] == p]
        idf = [i for _, _, o, _ in rs for i, _ in o]
        e2e = [i * d for _, _, o, _ in rs for i, d in o]
        lines.append(f"| {p} | {np.mean(idf):.1%} ({min(idf):.0%}-{max(idf):.0%}) | {np.mean(e2e):.1%} | "
                     f"{' '.join(f'{i:.0%}' for i in idf)} | {np.mean([x[3] for x in rs]):.1f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if a.md:
        Path(a.md).write_text(md)


if __name__ == "__main__":
    main()
