"""Time offsets between the cameras (GitHub issue #3).

  timing   frame timing of the 4K originals (container timestamps, no decoding): frames, rate, gaps
  shake    camera shake: the global image shift between consecutive frames of each 1080p video (phase correlation),
           cross-correlated between the two cameras on one mount (NL + FR on the left backboard, NR + FL on the
           right). A ball hitting the backboard shakes both cameras at the same moment, so the lag of the correlation
           peak is their time offset in frames.
  sheet    contact sheet of a camera's view of one rim over a frame range, to match a ball-rim contact by eye between
           a near camera and the far camera across the court (links the two mounts).
  agree    median distance between two cameras' MV3DT foot points of the same person (court-tracker tracks), at the
           same frame number and with the offsets applied: the offsets are right if the distance drops for every pair.

Offsets: the frame number at which each camera shows a given moment, minus NL's (negative = shows it earlier).

  python -m tools.sync_check timing
  python -m tools.sync_check shake [--end 11670]
  python -m tools.sync_check sheet FL left 8046 8099 out.jpg
  python -m tools.sync_check agree mount_r1 [mount_r2 ...] --offsets FL=-13,NR=-22,FR=-26
"""

import argparse
import collections
import itertools
import subprocess

import numpy as np

from tools import CAMERAS, GAME_END, INPUTS, ROOT

CAMERA_4K = {"NL": "35347", "FL": "35353", "NR": "38903", "FR": "35349"}   # matched by image content, 2026-10-27
MOUNTS = (("NL", "FR"), ("NR", "FL"))
HD = {pos: INPUTS / f"{stem}_1080p_15mins_cfr_undist.mp4" for _, pos, stem in CAMERAS}


def timing() -> None:
    for pos, num in CAMERA_4K.items():
        path = INPUTS / f"camera{num}_rtsp_20261019_214419.mp4"
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
                              "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
        pts = np.array(sorted(int(x) for x in out.split() if x.strip().isdigit()), float) / 90000
        d = np.diff(pts)
        gaps = np.where(d > 0.05)[0]
        print(f"{pos} (camera{num}): {len(pts)} frames over {pts[-1]:.2f} s, median interval {np.median(d) * 1000:.2f} ms "
              f"({1 / np.median(d):.2f} fps); gaps > 50 ms: "
              + (", ".join(f"{pts[i] // 60:.0f}:{pts[i] % 60:05.2f} ({d[i] * 1000:.0f} ms)" for i in gaps) or "none"))


def _shifts(path, end: int, w: int = 640, h: int = 360) -> np.ndarray:
    import cv2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(end + 1), "-vf", f"scale={w}:{h}",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    prev, out = None, []
    while True:
        b = p.stdout.read(w * h)
        if len(b) < w * h:
            break
        im = np.frombuffer(b, np.uint8).reshape(h, w).astype(np.float32)
        if prev is None:
            out.append(0.0)
        else:
            (dx, dy), r = cv2.phaseCorrelate(prev, im, win)
            out.append(float(np.hypot(dx, dy)) if r >= 0.1 else 0.0)
        prev = im
    return np.array(out)


def shake(end: int, maxlag: int = 40) -> None:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        sig = dict(zip(HD, ex.map(_shifts, HD.values(), [end] * 4)))
    for a, b in MOUNTS:
        x, y = sig[a] - sig[a].mean(), sig[b] - sig[b].mean()
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        c = [((x[:n - L] * y[L:]).sum() if L >= 0 else (x[-L:] * y[:n + L]).sum()) for L in range(-maxlag, maxlag + 1)]
        c = np.array(c) / np.sqrt((x * x).sum() * (y * y).sum())
        i = int(c.argmax())
        ev = lambda s: ", ".join(str(f) for f in np.where(s > max(1.0, np.percentile(s, 99.9)))[0][:8])
        print(f"{a}-{b}: correlation peak at lag {i - maxlag:+d} frames ({b} shows the moment {abs(i - maxlag)} frames "
              f"{'earlier' if i - maxlag < 0 else 'later'} than {a}), correlation {c[i]:.2f}; "
              f"largest shifts {a}: {ev(sig[a])}; {b}: {ev(sig[b])}")


def sheet(pos: str, end: str, f0: int, f1: int, out: str) -> None:
    from tools import DATASETS
    from tools.court import FT, HALF_LEFT_FT, HALF_RIGHT_FT
    from tools.stitch import load_cameras
    X = np.array([-(HALF_LEFT_FT - 5.25) * FT if end == "left" else (HALF_RIGHT_FT - 5.25) * FT, 0.0, 10 * FT, 1.0])
    p = load_cameras(DATASETS / "gym" / "camInfo")[pos] @ X
    u, v = p[:2] / p[2]
    near = (pos, end) in (("NL", "left"), ("NR", "right"))
    w, h = (560, 448) if near else (200, 160)
    l, t = int(np.clip(u - w / 2, 0, 1920 - w)) // 2 * 2, int(np.clip(v - h * 0.4, 0, 1080 - h)) // 2 * 2
    scale = "350:280" if near else "400:320"
    cols = 9
    rows = -(-(f1 - f0 + 1) // cols)
    vf = (f"select='between(n\\,{f0}\\,{f1})',crop={w}:{h}:{l}:{t},scale={scale},"
          f"drawtext=text='%{{n}}':x=5:y=5:fontsize=26:fontcolor=yellow:start_number={f0},tile={cols}x{rows}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(HD[pos]), "-vf", vf, "-frames:v", "1",
                    "-fps_mode", "passthrough", out], check=True)
    print(out)


def pair_distances(run, offsets: dict | None = None) -> dict:
    """(cam a, cam b) -> (distances at the same frame number, distances with b shifted by its offset relative to a):
    MV3DT foot points of the same person (court-tracker tracks) over the game."""
    from tools.gt_eval import game_end
    from tools.stitch import load_detections
    from tools.world_tracker import track_default
    offsets = offsets or {}
    last = game_end(run)
    d = load_detections(run / "kafka.jsonl.gz")
    pos = {(int(f), str(c), str(g)): (x, y) for f, c, g, x, y in zip(d["frame"], d["cam"], d["gid"], d["x"], d["y"])}
    res = collections.defaultdict(lambda: ([], []))
    for t in track_default(run):
        per = collections.defaultdict(dict)
        for k in t.dets:
            k = (int(k[0]), str(k[1]), str(k[2]))
            if k[0] <= last and k in pos:
                per[k[1]][k[0]] = pos[k]
        cams = [c for c in HD if c in per]
        for a, b in itertools.combinations(cams, 2):
            lag = offsets.get(b, 0) - offsets.get(a, 0)
            for f, pa in per[a].items():
                if f in per[b]:
                    res[(a, b)][0].append(np.hypot(*np.subtract(pa, per[b][f])))
                if f + lag in per[b]:
                    res[(a, b)][1].append(np.hypot(*np.subtract(pa, per[b][f + lag])))
    return dict(res)


def agree(runs: list[str], offsets: dict) -> None:
    for r in runs:
        res = pair_distances(ROOT / "experiments" / r, offsets)
        print(r + ": " + "; ".join(f"{a}-{b} {np.median(x):.2f} -> {np.median(y):.2f} m (n={len(x)})"
                                   for (a, b), (x, y) in sorted(res.items(), key=lambda kv: -len(kv[1][0]))
                                   if len(x) > 500))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("timing")
    s = sub.add_parser("shake")
    s.add_argument("--end", type=int, default=GAME_END)
    s = sub.add_parser("sheet")
    s.add_argument("pos", choices=list(HD))
    s.add_argument("end", choices=("left", "right"))
    s.add_argument("f0", type=int)
    s.add_argument("f1", type=int)
    s.add_argument("out")
    s = sub.add_parser("agree")
    s.add_argument("runs", nargs="+")
    s.add_argument("--offsets", required=True, help="e.g. FL=-13,NR=-22,FR=-26 (NL = 0)")
    a = ap.parse_args()
    if a.cmd == "timing":
        timing()
    elif a.cmd == "shake":
        shake(a.end)
    elif a.cmd == "sheet":
        sheet(a.pos, a.end, a.f0, a.f1, a.out)
    else:
        agree(a.runs, {k: int(v) for k, v in (x.split("=") for x in a.offsets.split(","))})


if __name__ == "__main__":
    main()
