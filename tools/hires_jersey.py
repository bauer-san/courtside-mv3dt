"""Jersey reads from the 4K recordings (GitHub issue #2): offline experiment, no DeepStream change.

DeepStream reads jersey numbers on the 1080p undistorted streams: for each player box at least 150 px high and not cut
by the image border a torso object (12-62 % of the box height, apps/court_filter/torso_objects.c), the PP-OCRv4 text
detector on it (480 x 384 input, letterboxed; re-inferred on the 7th frame per player) and the digit recogniser on
each of its top 3 text boxes (48 x 64; apps/jersey). This tool replays that path offline on a run's own boxes
(kafka.jsonl.gz), from two sources: the 1080p clip frames DeepStream saw (the baseline), and the same regions cut from
the 4K recordings at twice the resolution. Each result is a jersey-read file in the format of tools/jersey.py, for the
court tracker: world_tracker.track_default(run, reads_file=...).

Pixels: the 1080p videos were made from the 4K recordings (scaled by 1/2, then undistorted with the one-coefficient
radial model in inputs/*_distortion.yaml): the 1080p pixel q shows the 4K point 2 (c + (q - c)(1 + k1 |q - c|^2)),
c = image centre (median residual 0.2 px over 1,600-2,100 feature matches per frame, NL). A region of the 1080p frame is
cut from the 4K frame by sampling it on a grid twice as dense through that mapping.
Time: the constant 30 fps of the 1080p videos was resampled from the 4K timestamps, within a frame of the nearest
timestamp but by no fixed rule, so each 1080p frame is matched to the one of the 7 nearest 4K frames (by timestamp)
whose undistorted image differs least from it (at 960 x 540).

  python -m tools.hires_jersey read s_int8t10_r1 s_int8t10_r2 s_int8t10_r3 [--min-h 150,100] [--end FRAME]
      -> experiments/<run>/jersey_<1080p|4k>_h<min-h>.npz, experiments/hires_framemap_<dataset>.npz
  python -m tools.hires_jersey check s_int8t10_r1             # 1080p replay vs DeepStream's own reads (jersey.npz)
  python -m tools.hires_jersey score s_int8t10_r1 s_int8t10_r2 s_int8t10_r3 [--md out.md]
"""

import argparse
import collections
import gzip
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import yaml

from tools import CAMERAS, INPUTS, ROOT
from tools.jersey_rec_bench import decode
from tools.sync_check import CAMERA_4K

OCR = ROOT / "external/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/OCR"
DET_H, DET_W = 480, 384               # config_sgie_textdet.txt infer-dims
REC_H, REC_W = 48, 64                 # config_sgie_textrec.txt infer-dims
REINFER = 7                           # secondary-reinfer-interval=6: a player's torso is read on every 7th frame
T0, T1 = 0.12, 0.62                   # torso_objects.c
BIN_THRESH, BOX_THRESH, UNCLIP, MIN_AREA, MAX_BOXES = 0.3, 0.5, 1.6, 10, 3   # nvdsinfer_jersey_parsers.cpp
SOURCES = ("1080p", "4k")
CX, CY = 959.5, 539.5
AREA_DOWN = -1                        # --interp area


def _k1(stem: str) -> float:
    return float(yaml.safe_load((INPUTS / f"{stem}_1080p_15mins_cfr_distortion.yaml").read_text())["k1"])


def to_4k(x, y, k1: float):
    """Undistorted 1080p pixel coordinates -> 4K recording pixel coordinates."""
    dx, dy = x - CX, y - CY
    f = 1 + k1 * (dx * dx + dy * dy)
    return (CX + dx * f + 0.5) * 2 - 0.5, (CY + dy * f + 0.5) * 2 - 0.5


def _pts(path) -> np.ndarray:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    return np.array(sorted(int(x) for x in out.split() if x.strip().isdigit()), float) / 90000


def region(img: np.ndarray, box, scale: int, k1: float | None = None) -> np.ndarray:
    """Cut box (l, t, r, b, in undistorted 1080p pixels) from a 1080p frame (scale 1: integer crop, as nvinfer does)
    or from a 4K frame (scale 2, through the lens model)."""
    l, t, r, b = box
    if scale == 1:
        x0, y0 = int(max(0, l)), int(max(0, t))
        x1, y1 = int(min(img.shape[1], r)), int(min(img.shape[0], b))
        return img[y0:y1, x0:x1]
    l, t, r, b = max(0.0, l), max(0.0, t), min(1920.0, r), min(1080.0, b)
    w, h = max(1, round((r - l) * scale)), max(1, round((b - t) * scale))
    xs = l + (np.arange(w, dtype=np.float32) + 0.5) / scale - 0.5
    ys = t + (np.arange(h, dtype=np.float32) + 0.5) / scale - 0.5
    mx, my = to_4k(*np.meshgrid(xs, ys), k1)
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def letterbox(img: np.ndarray, H: int, W: int, symmetric: bool, interp: int):
    """nvinfer maintain-aspect-ratio scaling with black padding -> (network input, scale, pad x, pad y)."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(1, min(W, round(w * s))), max(1, min(H, round(h * s)))
    px, py = ((W - nw) // 2, (H - nh) // 2) if symmetric else (0, 0)
    out = np.zeros((H, W, 3), np.uint8)
    if interp == AREA_DOWN:     # area averaging when shrinking (no aliasing), bilinear when enlarging
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out, s, px, py


def _blob(ims: list) -> np.ndarray:
    return ((np.stack(ims).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2).copy()


def db_boxes(prob: np.ndarray) -> list:
    """Text boxes from a DB probability map, as NvDsInferParseCustomDBText: (l, t, r, b, score), best first."""
    H, W = prob.shape
    s = prob >= BIN_THRESH
    d = s.copy()
    d[:, 1:] |= s[:, :-1]
    d[1:, :] |= s[:-1, :]
    d[1:, 1:] |= s[:-1, :-1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(d.astype(np.uint8), connectivity=4)
    sums = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n)
    out = []
    for i in range(1, n):
        x0, y0, w, h, area = st[i]
        score = sums[i] / area
        if area < MIN_AREA or score < BOX_THRESH:
            continue
        off = w * h * UNCLIP / (2.0 * (w + h))
        l, t = max(0.0, x0 - off), max(0.0, y0 - off)
        r, b = min(float(W), x0 + w + off), min(float(H), y0 + h + off)
        if b - t >= 6:
            out.append((l, t, r, b, float(score)))
    return sorted(out, key=lambda x: -x[4])[:MAX_BOXES]


def _sessions():
    import onnxruntime as ort
    ort.preload_dlls()
    ort.set_default_logger_severity(3)
    prov = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
    return (ort.InferenceSession(str(OCR / "ch_PP-OCRv4_det_infer.onnx"), providers=prov),
            ort.InferenceSession(str(OCR / "ch_PP-OCRv4_rec_infer.onnx"), providers=prov))


def read_torsos(sess, jobs: list, interp: int) -> list:
    """jobs: (source image, scale, k1, torso box in 1080p pixels) -> per job the best digit read (number, conf) among
    its text boxes, ('', 0) if a text box but no read, None if no text box."""
    det, rec = sess
    if not jobs:
        return []
    torsos = [region(img, box, sc, k1) for img, sc, k1, box in jobs]
    lb = [letterbox(t, DET_H, DET_W, True, interp) for t in torsos]
    prob = np.concatenate([det.run(None, {"x": _blob([x[0] for x in lb[i:i + 32]])})[0]
                           for i in range(0, len(lb), 32)])[:, 0]
    crops, owner = [], []
    for j, ((img, sc, k1, (l, t, r, b)), (_, s, px, py)) in enumerate(zip(jobs, lb)):
        for bl, bt, br, bb, _ in db_boxes(prob[j]):
            box = (l + (bl - px) / s / sc, t + (bt - py) / s / sc, l + (br - px) / s / sc, t + (bb - py) / s / sc)
            c = region(img, box, sc, k1)
            if c.size:
                crops.append(letterbox(c, REC_H, REC_W, False, interp)[0])
                owner.append(j)
    res = [None] * len(jobs)
    if crops:
        out = np.concatenate([rec.run(None, {"x": _blob(crops[i:i + 64])})[0] for i in range(0, len(crops), 64)])
        for j, p in zip(owner, out):
            n, conf = decode(p)
            if res[j] is None or (n and conf > res[j][1]):
                res[j] = (n, conf if n else 0.0)
    return res


def load_boxes(run) -> dict:
    """cam -> frame -> list of (gid, l, t, w, h): the run's player boxes as DeepStream published them."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    with gzip.open(run / "kafka.jsonl.gz", "rt") as f:
        for line in f:
            m = json.loads(line)
            cam = _sensors().get(m["sensorId"], m["sensorId"])
            for o in m.get("objects", []):
                b = o.get("bbox", {})
                l, t = b.get("leftX", 0.0), b.get("topY", 0.0)
                out[cam][int(m["id"])].append((str(o["id"]), l, t, b.get("rightX", 0.0) - l, b.get("bottomY", 0.0) - t))
    return {c: dict(v) for c, v in out.items()}


def _sensors() -> dict:
    from tools.stitch import SENSORS
    return SENSORS


def schedule(frames: dict, min_h: float, end: int) -> dict:
    """frame -> [(gid, torso box)] that DeepStream's torso + reinfer rules send to the text detector."""
    last, out = {}, collections.defaultdict(list)
    for f in sorted(frames):
        if f > end:
            break
        for g, l, t, w, h in frames[f]:
            if h >= min_h and t > 5 and t + h < 1075 and l > 2 and l + w < 1918 and f - last.get(g, -REINFER) >= REINFER:
                last[g] = f
                out[f].append((g, (l, t + T0 * h, l + w, t + T1 * h)))
    return out


def _camera(args) -> tuple:
    """One camera: decode the clip and the 4K recording together, match frames, read every scheduled torso from both."""
    cam, stem, clip, start, end, scheds, interp = args
    k1 = _k1(stem)
    src4k = INPUTS / f"camera{CAMERA_4K[cam]}_rtsp_20261019_214419.mp4"
    pts = _pts(src4k)
    near = np.clip(np.searchsorted(pts, (np.arange(end + 1) + start) / 30.0), 1, len(pts) - 1)
    near -= (np.abs(pts[near - 1] - (np.arange(end + 1) + start) / 30.0) < np.abs(pts[near] - (np.arange(end + 1) + start) / 30.0))
    hd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", str(end + 1), "-fps_mode", "passthrough",
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    uhd = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", str(src4k), "-frames:v",       # every
                            str(int(near[-1]) + 5), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "nv12",  # frame
                            "-"], stdout=subprocess.PIPE)                                                    # once
    # comparison at 960 x 540: undistorted grid -> 4K/4 coordinates
    u, v = np.meshgrid(np.arange(960, dtype=np.float32), np.arange(540, dtype=np.float32))
    mx, my = to_4k((u + 0.5) * 2 - 0.5, (v + 0.5) * 2 - 0.5, k1)
    mx, my = ((mx + 0.5) / 4 - 0.5).astype(np.float32), ((my + 0.5) / 4 - 0.5).astype(np.float32)
    buf, k = collections.OrderedDict(), 0
    sess = _sessions()
    fmap = np.zeros((end + 1, 3), np.float32)
    results = collections.defaultdict(list)       # (run index, source, min_h) -> [(frame, gid, number, score)]
    n4k = 3840 * 2160 * 3 // 2
    for f in range(end + 1):
        raw = hd.stdout.read(1920 * 1080 * 3)
        if len(raw) < 1920 * 1080 * 3:
            break
        im = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)
        while k <= near[f] + 3:
            b = uhd.stdout.read(n4k)
            if len(b) < n4k:
                break
            y = np.frombuffer(b, np.uint8)
            small = cv2.remap(cv2.resize(y[:3840 * 2160].reshape(2160, 3840), (960, 540), interpolation=cv2.INTER_AREA),
                              mx, my, cv2.INTER_LINEAR).astype(np.float32)
            small = ((small - 16) * (255 / 219)).astype(np.int16)        # video-range luma -> the grey of a BGR frame
            buf[k] = (small, y)
            k += 1
        for j in [j for j in buf if j < near[f] - 3]:
            del buf[j]
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (960, 540), interpolation=cv2.INTER_AREA).astype(np.int16)
        cand = sorted((float(np.abs(s[20:-20, 20:-20] - g[20:-20, 20:-20]).mean()), j)
                      for j, (s, _) in buf.items() if abs(j - near[f]) <= 3)
        if not cand:
            break
        fmap[f] = (cand[0][1], cand[0][0], cand[1][0] - cand[0][0] if len(cand) > 1 else 0)
        todo = [(ri, mh, g_, box) for (ri, mh), s in scheds.items() for g_, box in s.get(f, [])]
        if not todo:
            continue
        im4k = cv2.cvtColor(buf[cand[0][1]][1].reshape(2160 * 3 // 2, 3840), cv2.COLOR_YUV2BGR_NV12)
        uniq = sorted({(ri, g_, box) for ri, _, g_, box in todo})
        idx = {u_: i for i, u_ in enumerate(uniq)}
        for src, img, sc in (("1080p", im, 1), ("4k", im4k, 2)):
            res = read_torsos(sess, [(img, sc, k1, box) for _, _, box in uniq], interp)
            for ri, mh, g_, box in todo:
                r = res[idx[(ri, g_, box)]]
                if r is not None:
                    results[(ri, src, mh)].append((f, g_, r[0], r[1]))
    hd.kill()
    uhd.kill()
    return cam, fmap, dict(results)


def read(runs: list, min_hs: list, end: int | None, interp: int, tag: str = "") -> None:
    from tools.appearance import run_dataset
    from tools.gt_eval import game_end
    paths = [ROOT / "experiments" / r for r in runs]
    ds = {run_dataset(p) for p in paths}
    assert len(ds) == 1, "runs must share one dataset (the same clip frames)"
    ds = ds.pop()
    man = json.loads((ds / "manifest.json").read_text())
    end = end if end is not None else game_end(paths[0])
    with ProcessPoolExecutor(len(paths)) as ex:
        boxes = list(ex.map(load_boxes, paths))
    jobs = []
    for cid, pos, stem in CAMERAS:
        scheds = {(ri, mh): schedule(b.get(pos, {}), mh, end) for ri, b in enumerate(boxes) for mh in min_hs}
        jobs.append((pos, stem, ds / "videos" / f"{cid}_{pos}.mp4", man["cameras"][cid]["start_frame"], end, scheds, interp))
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(_camera, jobs))
    np.savez_compressed(ROOT / "experiments" / f"hires_framemap_{ds.name}.npz", **{c: m for c, m, _ in out})
    for c, m, _ in out:
        print(f"{c}: 4K frame per clip frame matched; median difference {np.median(m[:, 1]):.2f}, to the runner-up "
              f"{np.median(m[:, 2]):.2f} (grey levels at 960 x 540)")
    for ri, p in enumerate(paths):
        for src in SOURCES:
            for mh in min_hs:
                rows = sorted((f, c, g, n, s) for c, _, res in out for f, g, n, s in res.get((ri, src, mh), []))
                name = f"jersey_{src}_h{mh:g}{tag}.npz"
                np.savez_compressed(p / name, frame=np.array([r[0] for r in rows]), cam=np.array([r[1] for r in rows]),
                                    gid=np.array([r[2] for r in rows]), number=np.array([r[3] for r in rows]),
                                    score=np.array([r[4] for r in rows], np.float32))
                print(f"{p.name}/{name}: {len(rows)} torsos with text, {sum(1 for r in rows if r[3])} reads")


def check(run: str, end: int | None) -> None:
    """How closely the offline 1080p replay reproduces DeepStream's own reads (same boxes, same frames)."""
    from tools.gt_eval import game_end
    from tools.jersey import load
    p = ROOT / "experiments" / run
    end = end if end is not None else game_end(p)
    ds, py = load(p, "jersey.npz"), load(p, "jersey_1080p_h150.npz")
    D = {(int(f), str(c), str(g)): str(n) for f, c, g, n in zip(ds["frame"], ds["cam"], ds["gid"], ds["number"]) if f <= end}
    P = {(int(f), str(c), str(g)): str(n) for f, c, g, n in zip(py["frame"], py["cam"], py["gid"], py["number"]) if f <= end}
    both = D.keys() & P.keys()
    rd, rp = {k for k, n in D.items() if n}, {k for k, n in P.items() if n}
    same = sum(1 for k in rd & rp if D[k] == P[k])
    print(f"{run}: torsos with text DeepStream {len(D)}, replay {len(P)}, both {len(both)}; reads DeepStream {len(rd)}, "
          f"replay {len(rp)}, both {len(rd & rp)} of which the same number {same}")


def read_accuracy(run, name: str) -> dict:
    """Reads near a labelled moment (same camera and MV3DT id, within 15 frames): correct / wrong number."""
    from tools.gt_eval import frame_offsets, load_gt
    from tools.jersey import load
    off = frame_offsets(run)
    labels = collections.defaultdict(list)
    for n in ("game_2to3", "game_4to5"):
        for (m, c, g), ident in load_gt(n, run).items():
            labels[(c, g)].append((m, ident.split(":")[1]))
    j = load(run, name)
    out = collections.Counter()
    for f, c, g, num in zip(j["frame"], j["cam"], j["gid"], j["number"]):
        if not num:
            continue
        near = [(abs(m - (int(f) + off.get(str(c), 0))), ident) for m, ident in labels.get((str(c), str(g)), [])]
        near = [x for x in near if x[0] <= 15]
        if near and min(near)[1].isdigit():
            out["correct" if min(near)[1] == str(num) else "wrong"] += 1
    return out


def identity(run, name: str) -> dict:
    """Court tracker identity IDF1 and end-to-end share per labelled minute with the given read file."""
    from tools.gt_eval import frame_offsets, load_gt, score
    from tools.sync_eval import MINUTES, _unshift
    from tools.world_tracker import identity_tracks, track_default, track_run
    assert (run / name).exists(), run / name
    off = frame_offsets(run)
    G = _unshift(identity_tracks(track_default(run, reads_file=name), track_run.last_det_ident), off)
    out = {}
    for n in MINUTES:
        gt = {(f + off.get(c, 0), c, g): v for (f, c, g), v in load_gt(n, run).items()}
        s = score(G, gt)
        out[n] = (s["idf1"], s["idf1"] * len(gt) / len(load_gt(n)), s["switches"])
    return out


def _score_one(a):
    run, name = a
    from tools.gt_eval import game_end
    from tools.jersey import load
    p = ROOT / "experiments" / run
    j = load(p, name)
    g = j["frame"] <= game_end(p)
    return run, name, int(g.sum()), int((g & (j["number"] != "")).sum()), read_accuracy(p, name), identity(p, name)


def score_all(runs: list, files: list, md: str | None) -> None:
    with ProcessPoolExecutor(min(8, len(runs) * len(files))) as ex:
        res = list(ex.map(_score_one, [(r, f) for f in files for r in runs]))
    lines = ["| reads from | runs | torsos with text (game) | reads (game) | reads near labels: correct / wrong | "
             "court tracker IDF1 | end-to-end | switches per minute |", "|---|---|---|---|---|---|---|---|"]
    for f in files:
        rs = [r for r in res if r[1] == f]
        idf = [r[5][n][0] for r in rs for n in r[5]]
        e2e = [r[5][n][1] for r in rs for n in r[5]]
        sw = [r[5][n][2] for r in rs for n in r[5]]
        c, w = sum(r[4]["correct"] for r in rs), sum(r[4]["wrong"] for r in rs)
        lines.append(f"| {f} | {len(rs)} | {np.mean([r[2] for r in rs]):.0f} | {np.mean([r[3] for r in rs]):.0f} | "
                     f"{c} / {w} | {np.mean(idf):.1%} ({min(idf):.0%}-{max(idf):.0%}) | {np.mean(e2e):.1%} | "
                     f"{np.mean(sw):.0f} |")
    lines += ["", "Per run and minute (court tracker IDF1): " + "; ".join(
        f"{r[1]} {r[0]} " + " ".join(f"{v[0]:.0%}" for v in r[5].values()) for r in res)]
    text = "\n".join(lines) + "\n"
    print(text)
    if md:
        open(md, "w").write(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read", "check", "score"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--min-h", default="150,100", help="read: minimum player box heights (1080p px)")
    ap.add_argument("--end", type=int, help="last clip frame (default: the end of the game)")
    ap.add_argument("--interp", choices=["nearest", "linear", "area"], default="linear",
                    help="read: network input scaling (area: area averaging when shrinking, bilinear when enlarging)")
    ap.add_argument("--tag", default="", help="read: suffix of the output file names")
    ap.add_argument("--files", default="jersey.npz,jersey_1080p_h150.npz,jersey_4k_h150.npz,jersey_1080p_h100.npz,"
                                        "jersey_4k_h100.npz", help="score: read files to compare")
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "read":
        read(a.runs, [float(x) for x in a.min_h.split(",")], a.end,
             {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "area": AREA_DOWN}[a.interp], a.tag)
    elif a.cmd == "check":
        for r in a.runs:
            check(r, a.end)
    else:
        score_all(a.runs, a.files.split(","), a.md)


if __name__ == "__main__":
    main()
