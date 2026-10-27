"""Appearance cues for the court-plane tracker: team colour and body height.

extract   decode each camera's video, sample every `step`-th frame, and for every recorded detection
          (kafka.jsonl.gz bbox) compute
            - torso colour: HSV hue/saturation histogram (18 x 8) of the 20-50 % height, central 50 % width
              crop of the box
            - height (m): back-project the bbox bottom onto the floor, then find the height whose
              projection meets the bbox top (only if the box is not cut by the image border); the steep
              backboard cameras (NL, NR) are excluded, their boxes rarely contain the whole person
          -> experiments/<run>/appearance.npz  (frame, cam, gid, hist, height, bbox, cut)
             cut = box touches the image border (person partly outside the frame)
teams     k-means (k clusters) on sqrt(histogram) of good crops only (not cut, >= MIN_BOX_H px, not from the
          steep NL camera, whose boxes are mostly legs), centred per camera (removes camera colour casts);
          writes a contact sheet
          (team_clusters.jpg) and teams.yml mapping cluster -> label. Labels are guessed (the two largest
          clusters are teams A and B, the rest "other") and MUST be checked on the sheet and edited
          (`set-teams`): team colours change from game to game and the size guess is often wrong. Any label
          other than "other" is a team; several clusters may share a label.
Kafka frame ids are DeepStream frame_num (0-based): frame f is video frame index f (checked against the
jersey_reads.csv probe, 2026-10-22; earlier extractions used f - 1, i.e. one frame early).

  python -m tools.appearance extract <run> [--step 3]
  python -m tools.appearance teams <run> [--k 4] [--max-frame GAME_END]   # colour cues only from game frames
  python -m tools.appearance set-teams <run> 0=red 1=other 2=maroon ...
"""

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from tools import CALIB_HEIGHT, CALIB_WIDTH, CAMERAS, DATASETS, GAME_END, ROOT
from tools.stitch import SENSORS, floor_point, load_cameras

HEIGHT_CAMS = {"FL", "FR"}
COLOUR_CAMS = {"FL", "FR", "NR"}
MIN_BOX_H = 80
LABEL_P = 0.8          # label only if the soft assignment is at least this probable ...
LABEL_DIST_Q = 80      # ... and the sample is within this percentile of its cluster's distances
TEAM_LABELS = ("A", "B")


def run_dataset(run: Path) -> Path:
    m = re.search(r"dataset=(\S+)", (run / "run.txt").read_text())
    d = DATASETS / m.group(1)
    if not d.exists():   # runs made before the datasets were renamed (courtside* -> gym*, 2026-10-25)
        d = DATASETS / m.group(1).replace("courtside", "gym")
    return d


def load_boxes(run: Path, step: int) -> dict:
    """(cam, frame) -> list of (gid, l, t, r, b) for frames divisible by step."""
    boxes = collections.defaultdict(list)
    with gzip.open(run / "kafka.jsonl.gz", "rt") as f:
        for line in f:
            m = json.loads(line)
            fr = int(m["id"])
            if fr % step:
                continue
            cam = SENSORS.get(m["sensorId"], m["sensorId"])
            for o in m.get("objects", []):
                b = o.get("bbox", {})
                boxes[(cam, fr)].append((o["id"], b.get("leftX", 0.0), b.get("topY", 0.0),
                                         b.get("rightX", 0.0), b.get("bottomY", 0.0)))
    return boxes


def torso_hist(img: np.ndarray, l: int, t: int, r: int, b: int) -> np.ndarray | None:
    h, w = b - t, r - l
    if h < 60 or w < 15:
        return None
    tor = img[t + int(0.2 * h): t + int(0.5 * h), l + int(0.25 * w): r - int(0.25 * w)]
    if tor.size == 0:
        return None
    hsv = cv2.cvtColor(tor, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).ravel()
    return hist / (hist.sum() + 1e-9)


def person_height(P: np.ndarray, l: float, t: float, r: float, b: float) -> float:
    if t <= 5 or b >= CALIB_HEIGHT - 5 or l <= 2 or r >= CALIB_WIDTH - 2:
        return np.nan
    xy, ok = floor_point(P, np.array([(l + r) / 2]), np.array([b]))
    if not ok[0]:
        return np.nan
    hs = np.linspace(0.5, 2.6, 211)
    pts = np.c_[np.full(len(hs), xy[0, 0]), np.full(len(hs), xy[0, 1]), hs, np.ones(len(hs))] @ P.T
    v = pts[:, 1] / pts[:, 2]
    i = int(np.argmin(np.abs(v - t)))
    return float(hs[i]) if 0 < i < len(hs) - 1 else np.nan


def extract(run: Path, step: int = 3) -> Path:
    ds = run_dataset(run)
    cams = load_cameras(run / "camInfo_used")
    boxes = load_boxes(run, step)
    rows = collections.defaultdict(list)
    for cam_id, pos, _ in CAMERAS:
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        idx = 0
        while True:
            fr = idx
            if fr % step == 0 and (pos, fr) in boxes:
                ok, img = cap.read()
            else:
                ok = cap.grab()
                img = None
            if not ok:
                break
            idx += 1
            if img is None:
                continue
            for gid, l, t, r, b in boxes[(pos, fr)]:
                li, ti, ri, bi = int(max(0, l)), int(max(0, t)), int(min(CALIB_WIDTH - 1, r)), int(min(CALIB_HEIGHT - 1, b))
                hist = torso_hist(img, li, ti, ri, bi)
                if hist is None:
                    continue
                rows["frame"].append(fr)
                rows["cam"].append(pos)
                rows["gid"].append(gid)
                rows["hist"].append(hist.astype(np.float16))
                rows["height"].append(person_height(cams[pos], l, t, r, b) if pos in HEIGHT_CAMS else np.nan)
                rows["bbox"].append((l, t, r, b))
                rows["cut"].append(t <= 5 or b >= CALIB_HEIGHT - 5 or l <= 2 or r >= CALIB_WIDTH - 2)
        cap.release()
        print(f"{pos}: {sum(1 for c in rows['cam'] if c == pos)} samples", flush=True)
    out = run / "appearance.npz"
    np.savez_compressed(out, **{k: np.array(v) for k, v in rows.items()})
    return out


def load_appearance(run: Path) -> dict:
    """All arrays in memory (indexing an NpzFile re-reads and decompresses the array on every access)."""
    with np.load(run / "appearance.npz") as z:
        return {k: z[k] for k in z.files}


def good_crops(a: dict) -> np.ndarray:
    h = a["bbox"][:, 3] - a["bbox"][:, 1]
    return (~a["cut"]) & (h >= MIN_BOX_H) & np.isin(a["cam"], list(COLOUR_CAMS))


def features(a: dict, cam_mean: dict) -> np.ndarray:
    """sqrt-histogram centred per camera (camera colour casts removed), plus the global mean."""
    X = np.sqrt(a["hist"].astype(np.float32))
    out = X.copy()
    for cam, m in cam_mean.items():
        sel = a["cam"] == cam
        out[sel] = X[sel] - m + cam_mean["_global"]
    return out


def cluster_teams(run: Path, k: int = 4, seed: int = 0, max_frame: int | None = None) -> dict:
    a = load_appearance(run)
    sel = good_crops(a)
    if max_frame is not None:
        sel &= a["frame"] <= max_frame
    a = {key: v[sel] for key, v in a.items()}
    X0 = np.sqrt(a["hist"].astype(np.float32))
    cam_mean = {c: X0[a["cam"] == c].mean(0) for c in np.unique(a["cam"])}
    cam_mean["_global"] = X0.mean(0)
    X = features(a, cam_mean)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-5)
    cv2.setRNGSeed(seed)
    _, lab, cent = cv2.kmeans(X, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    sizes = np.bincount(lab, minlength=k)
    order = np.argsort(-sizes)
    mapping = {int(c): (TEAM_LABELS[i] if i < 2 else "other") for i, c in enumerate(order)}
    d = np.sqrt(((X[:, None, :] - cent[None]) ** 2).sum(-1))[np.arange(len(X)), lab]
    radius = np.array([np.percentile(d[lab == c], LABEL_DIST_Q) if (lab == c).any() else np.inf for c in range(k)])
    np.savez(run / "team_model.npz", centres=cent, radius=radius,
             cams=np.array([c for c in cam_mean if c != "_global"]),
             cam_means=np.array([cam_mean[c] for c in cam_mean if c != "_global"]), global_mean=cam_mean["_global"])
    (run / "teams.yml").write_text(
        "# cluster -> label (A, B or other). Guessed from cluster sizes: CHECK team_clusters.jpg and edit.\n"
        + yaml.dump({"confirmed": False, "max_frame": max_frame, "mapping": mapping}, default_flow_style=False))
    contact_sheet(run, a, lab, mapping)
    return mapping


def contact_sheet(run: Path, a, lab: np.ndarray, mapping: dict, per: int = 30) -> None:
    ds = run_dataset(run)
    rng = np.random.default_rng(0)
    pick = {c: rng.choice(np.where(lab == c)[0], min(per, int((lab == c).sum())), replace=False)
            for c in mapping}
    need = collections.defaultdict(list)  # (cam, frame) -> [(cluster, gid)]
    for c, idx in pick.items():
        for i in idx:
            need[(str(a["cam"][i]), int(a["frame"][i]))].append((c, str(a["gid"][i])))
    boxes = load_boxes(run, 1)
    crops = collections.defaultdict(list)
    for cam_id, pos, _ in CAMERAS:
        frames = sorted(f for (c, f) in need if c == pos)
        cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{pos}.mp4").resolve()))
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            bx = {g: (l, t, r, b) for g, l, t, r, b in boxes[(pos, fr)]}
            for c, g in need[(pos, fr)]:
                if g in bx:
                    l, t, r, b = (int(max(0, v)) for v in bx[g])
                    crops[c].append(cv2.resize(img[t:b, l:r], (48, 96)))
        cap.release()
    rows = []
    for c in sorted(mapping, key=lambda c: (mapping[c] == "other", mapping[c], c)):
        tiles = crops[c][:per] + [np.full((96, 48, 3), 255, np.uint8)] * (per - len(crops[c][:per]))
        strip = np.hstack(tiles)
        head = np.full((26, strip.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"cluster {c} -> {mapping[c]}  (n={int((lab == c).sum())})", (5, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        rows += [head, strip]
    cv2.imwrite(str(run / "team_clusters.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def team_probs(run: Path, temperature: float = 0.05) -> dict:
    """Only good crops in the configured frame range get probabilities; outliers get all zeros."""
    """(frame, cam, gid) -> {label: probability} from soft assignment to the cluster centres."""
    a = load_appearance(run)
    model = np.load(run / "team_model.npz")
    cent, radius = model["centres"], model["radius"]
    cam_mean = {str(c): m for c, m in zip(model["cams"], model["cam_means"])}
    cam_mean["_global"] = model["global_mean"]
    cfg = yaml.safe_load((run / "teams.yml").read_text())
    mapping = cfg["mapping"]
    sel = good_crops(a)
    if cfg.get("max_frame"):
        # colour cues are only valid while the clustered teams are on court (e.g. the game, not a warm-up)
        sel &= a["frame"] <= cfg["max_frame"]
    a = {key: v[sel] for key, v in a.items()}
    X = features(a, cam_mean)
    d = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
    w = np.exp(-(d - d.min(1, keepdims=True)) / temperature)
    w /= w.sum(1, keepdims=True)
    best = d.argmin(1)
    far = np.sqrt(d[np.arange(len(X)), best]) > radius[best]   # outlier for its cluster: no label
    w[far] = 0.0
    labels = sorted(set(mapping.values()))  # probabilities below LABEL_P are discarded by the tracker
    out = {}
    for i in range(len(X)):
        p = collections.defaultdict(float)
        for c, lab in mapping.items():
            p[lab] += float(w[i, int(c)])
        out[(int(a["frame"][i]), str(a["cam"][i]), str(a["gid"][i]))] = (
            {lab: p[lab] for lab in labels}, float(a["height"][i]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "teams", "set-teams"])
    ap.add_argument("run")
    ap.add_argument("mapping", nargs="*", help="set-teams: cluster=label ...")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-frame", type=int, default=None, help="use only samples up to this kafka frame")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    if args.cmd == "extract":
        print(extract(run, args.step))
    elif args.cmd == "set-teams":
        mapping = {int(k): v for k, v in (m.split("=") for m in args.mapping)}
        old = yaml.safe_load((run / "teams.yml").read_text()) if (run / "teams.yml").exists() else {}
        (run / "teams.yml").write_text(yaml.dump({"confirmed": True, "max_frame": old.get("max_frame"),
                                                  "mapping": mapping}, default_flow_style=False))
        print(mapping)
    else:
        print(cluster_teams(run, args.k, max_frame=args.max_frame), "-> check", run / "team_clusters.jpg")


if __name__ == "__main__":
    main()
