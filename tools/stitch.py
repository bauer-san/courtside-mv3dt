"""Offline clean-up of MV3DT output: court filter, duplicate merge, tracklet stitching.

Steps (all in world metres, using the recorded Kafka messages):
  1. ground filter   drop detections whose bbox bottom-centre, back-projected onto the floor (z = 0)
                     through that camera's matrix, lies outside the court + margin (benches, scorer's
                     table, crowd). Boxes touching the bottom image edge are kept (feet not visible).
                     MV3DT's own footLocation3D cannot be used for this: for seated / occluded people
                     the person model places them metres inside the court.
  2. fuse            per global ID and frame, average the cameras' foot positions
  3. merge           IDs that overlap in time and stay within `dup_dist` for most of the overlap are the
                     same person (cross-camera association MV3DT missed)
  4. stitch          link a track end to a later track start (gap <= max_gap) if the jump is plausible
                     (distance <= d0 + vmax * gap); greedy by normalised cost, one-to-one
  5. drop            stitched tracks shorter than min_life
"""

import collections
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SENSORS = {"Camera1": "NL", "Camera2": "FL", "Camera3": "NR", "Camera4": "FR"}  # msgconv: sorted video order
FPS = 30


@dataclass
class Params:
    margin: float = 0.75       # m beyond the court lines still counted as on court
    frame_h: int = 1080
    dup_dist: float = 0.6      # m
    dup_min_overlap: int = 15  # frames
    dup_frac: float = 0.8
    max_gap: int = 60          # frames
    d0: float = 1.0            # m
    vmax: float = 7.0          # m/s
    min_life: int = 30         # frames


@dataclass
class Track:
    frames: np.ndarray
    xy: np.ndarray
    cams: set = field(default_factory=set)
    members: list = field(default_factory=list)

    @property
    def start(self):
        return int(self.frames[0])

    @property
    def end(self):
        return int(self.frames[-1])


def load_detections(path: Path) -> dict:
    """Flat arrays: frame, cam, gid, x, y (MV3DT foot), u, v (bbox bottom-centre pixel), top (bbox top y)."""
    cols = collections.defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            m = json.loads(line)
            fr, cam = int(m["id"]), SENSORS.get(m["sensorId"], m["sensorId"])
            for o in m.get("objects", []):
                fl = o.get("info", {}).get("footLocation3D")
                if not fl:
                    continue
                b = o.get("bbox", {})  # protobuf JSON omits zero-valued fields
                x, y = (float(v) for v in fl.split(","))
                for k, v in (("frame", fr), ("cam", cam), ("gid", o["id"]), ("x", x), ("y", y),
                             ("u", (b.get("leftX", 0.0) + b.get("rightX", 0.0)) / 2), ("v", b.get("bottomY", 0.0)),
                             ("top", b.get("topY", 0.0))):
                    cols[k].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def load_cameras(caminfo_dir: Path) -> dict:
    out = {}
    for p in sorted(caminfo_dir.glob("cam_*.yml")):
        pos = p.stem.split("_")[-1]
        out[pos] = np.array(yaml.safe_load(p.read_text())["projectionMatrix_3x4_w2p"], float).reshape(3, 4)
    return out


def floor_point(P: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project pixels onto z = 0. Returns (N, 2) world xy and a validity mask (in front of camera)."""
    H = P[:, [0, 1, 3]]
    X = np.linalg.solve(H, np.vstack([u, v, np.ones_like(u)]))
    xy = (X[:2] / X[2]).T
    depth = (P @ np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))].T)[2]
    return xy, depth > 0


def ground_filter(d: dict, cams: dict, inside_fn, p: Params) -> np.ndarray:
    keep = np.ones(len(d["frame"]), bool)
    for pos, P in cams.items():
        sel = d["cam"] == pos
        if not sel.any():
            continue
        xy, ok = floor_point(P, d["u"][sel], d["v"][sel])
        on = ok & inside_fn(xy, p.margin)
        cut = d["v"][sel] >= p.frame_h - 5  # feet below the image edge: unknown, keep
        keep[sel] = on | cut
    return keep


def fuse(d: dict, mask: np.ndarray) -> dict[str, Track]:
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    cams = collections.defaultdict(set)
    for fr, cam, gid, x, y in zip(d["frame"][mask], d["cam"][mask], d["gid"][mask], d["x"][mask], d["y"][mask]):
        acc[gid][int(fr)].append((x, y))
        cams[gid].add(cam)
    tracks = {}
    for gid, per in acc.items():
        fr = np.array(sorted(per))
        tracks[gid] = Track(fr, np.array([np.mean(per[f], axis=0) for f in fr]), cams[gid], [gid])
    return tracks


class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, k):
        while self.p[k] != k:
            self.p[k] = self.p[self.p[k]]
            k = self.p[k]
        return k

    def union(self, a, b):
        self.p[self.find(b)] = self.find(a)


def _combine(tracks: dict[str, Track], groups: dict) -> dict[str, Track]:
    out = {}
    for root, keys in groups.items():
        per = collections.defaultdict(list)
        cams, members = set(), []
        for k in keys:
            t = tracks[k]
            for f, xy in zip(t.frames, t.xy):
                per[int(f)].append(xy)
            cams |= t.cams
            members += t.members
        fr = np.array(sorted(per))
        out[root] = Track(fr, np.array([np.mean(per[f], axis=0) for f in fr]), cams, members)
    return out


def merge_duplicates(tracks: dict[str, Track], p: Params) -> dict[str, Track]:
    keys = sorted(tracks, key=lambda k: tracks[k].start)
    uf = _UF(keys)
    active = []
    for k in keys:
        t = tracks[k]
        active = [a for a in active if tracks[a].end >= t.start]
        for a in active:
            o = tracks[a]
            common, i1, i2 = np.intersect1d(o.frames, t.frames, return_indices=True)
            if len(common) < p.dup_min_overlap:
                continue
            close = np.linalg.norm(o.xy[i1] - t.xy[i2], axis=1) < p.dup_dist
            if close.mean() >= p.dup_frac:
                uf.union(a, k)
        active.append(k)
    groups = collections.defaultdict(list)
    for k in keys:
        groups[uf.find(k)].append(k)
    return _combine(tracks, groups)


def stitch(tracks: dict[str, Track], p: Params) -> dict[str, Track]:
    keys = sorted(tracks, key=lambda k: tracks[k].start)
    starts = np.array([tracks[k].start for k in keys])
    cand = []
    for a in keys:
        ta = tracks[a]
        end_xy = ta.xy[-5:].mean(axis=0)
        lo, hi = np.searchsorted(starts, ta.end + 1), np.searchsorted(starts, ta.end + p.max_gap, side="right")
        for b in keys[lo:hi]:
            tb = tracks[b]
            gap = tb.start - ta.end
            dist = float(np.linalg.norm(tb.xy[:5].mean(axis=0) - end_xy))
            allow = p.d0 + p.vmax * gap / FPS
            if dist <= allow:
                cand.append((dist / allow + gap / p.max_gap, a, b))
    cand.sort()
    succ, pred = {}, {}
    for _, a, b in cand:
        if a not in succ and b not in pred:
            succ[a], pred[b] = b, a
    groups = {}
    for k in keys:
        if k in pred:
            continue
        chain, cur = [k], k
        while cur in succ:
            cur = succ[cur]
            chain.append(cur)
        groups[k] = chain
    return _combine(tracks, groups)


def clean(run: Path, inside_fn, p: Params | None = None) -> tuple[dict[str, Track], dict]:
    """Full pipeline for one run. Returns stitched tracks and step counts."""
    p = p or Params()
    d = load_detections(run / "kafka.jsonl.gz")
    cams = load_cameras(run / "camInfo_used")
    keep = ground_filter(d, cams, inside_fn, p)
    t0 = fuse(d, np.ones(len(keep), bool))
    t1 = fuse(d, keep)
    t2 = merge_duplicates(t1, p)
    t3 = stitch(t2, p)
    t4 = {k: t for k, t in t3.items() if t.end - t.start + 1 >= p.min_life}
    steps = {"detections": int(len(keep)), "kept_detections": int(keep.sum()),
             "ids_raw": len(t0), "ids_on_court": len(t1), "ids_merged": len(t2),
             "ids_stitched": len(t3), "ids_final": len(t4)}
    return t4, steps


def save_tracks(tracks: dict[str, Track], path: Path) -> None:
    with path.open("w") as f:
        f.write("track,frame,x,y,n_members\n")
        for i, (k, t) in enumerate(sorted(tracks.items(), key=lambda kv: kv[1].start)):
            for fr, (x, y) in zip(t.frames, t.xy):
                f.write(f"{i},{fr},{x:.3f},{y:.3f},{len(t.members)}\n")
