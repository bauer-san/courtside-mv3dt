"""Online, unsupervised team segmentation: two teams, referees and others from torso colour, without an operator.

Causal: the labels given to detections of frame f come from a model built from frames before f.

  colour      torso HSV histogram (appearance.torso_hist) of good crops, sqrt, centred per camera with running means
              (camera colour casts removed as they are learned)
  prototypes  each court track keeps its recent good-crop samples; a track's prototype is their mean
  clustering  every `every` frames, weighted k-means (weight = samples) on the prototypes of tracks seen in the last
              `window` frames; the first model from 10 weighted k-means++ starts, later ones from the previous
              centres
  roles       teams: the two clusters whose tracks get the most jersey reads per detection (players' numbers are
              read, referees' and spectators' hardly ever); referees: the remaining cluster selected by
              `ref_rule`; the rest: other. Team names A/B follow their cluster from one clustering to the next
              (centre matching), so identities (team:number) stay stable.
  labels      soft assignment of a crop to the centres (as appearance.team_probs); a label needs p >= LABEL_P and
              the crop within its cluster's 80th-percentile radius
Before `warmup` frames and `min_tracks` usable tracks there is no model and no labels.

  python -m tools.online_teams mount_r1 [mount_r2 ...]     # identity IDF1 with online labels vs the offline model
"""

import argparse
import collections
import math
from dataclasses import dataclass

import numpy as np

UNKNOWN = "other"


def _ssum(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sequential sum along an axis (cumsum), not NumPy's pairwise or BLAS summation, so that the C++ port
    (apps/courttrack/online_teams.cpp, plain loops) gives bit-identical results."""
    return np.take(np.cumsum(a, axis=axis), -1, axis=axis)


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared distances (float64, summed in order) between rows of X (n, D) and rows of C (k, D)."""
    return _ssum((X.astype(np.float64)[:, None] - np.asarray(C, np.float64)[None]) ** 2)


@dataclass
class OTParams:
    k: int = 3                 # clusters: two teams, everyone else
    every: int = 150           # frames between re-clusterings (5 s)
    window: int = 5400         # frames of track history used (3 min)
    warmup: int = 900          # frames before the first model (30 s)
    min_tracks: int = 8        # tracks with enough samples before the first model
    min_samples: int = 10      # good crops for a track to count
    track_recent: int = 900    # frames of a track's own samples in its prototype (30 s)
    temperature: float = 0.05
    label_p: float = 0.8
    radius_q: float = 80
    min_team_share: float = 0.1   # a team cluster carries at least this share of the samples
    seed: int = 0
    warm_only: bool = True     # after the first model, re-cluster only from the previous centres (no restarts):
                               # 79.2-80.2 % over 5 seeds, vs 78.5-80.7 % with 10 restarts every time
    ref_rule: str = "low_reads"   # low_reads: fewest reads per detection among the non-team clusters
    ref_label: str = "other"      # label of that cluster ("other": no label; "ref" was tested with k = 4)


class SplitMix64:
    """Small deterministic generator (same sequence in the C++ port)."""
    def __init__(self, seed: int = 0):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self) -> float:
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def choice(self, p: np.ndarray) -> int:
        """Index drawn with probabilities proportional to p (cumulative sum, as in the C++ port)."""
        c = np.cumsum(p)
        return int(min(np.searchsorted(c, self.uniform() * c[-1], side="right"), len(p) - 1))


def _lloyd(X, w, c, iters):
    for _ in range(iters):
        lab = _sqdist(X, c).argmin(1)
        nc = np.array([_ssum(X[lab == j] * w[lab == j, None], axis=0) / _ssum(w[lab == j]) if (lab == j).any()
                       else c[j] for j in range(len(c))])
        if np.array_equal(nc, c):
            break
        c = nc
    lab = _sqdist(X, c).argmin(1)
    return c, lab, float(_ssum(_ssum((X - c[lab]) ** 2) * w))


def weighted_kmeans(X: np.ndarray, w: np.ndarray, k: int, rng: SplitMix64, init: np.ndarray | None = None,
                    iters: int = 100, attempts: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means: `attempts` weighted k-means++ starts plus, if given, a start from `init` (the previous
    clustering's centres); the lowest weighted inertia wins."""
    X = X.astype(np.float64)
    starts = []
    for _ in range(attempts):
        c = [X[rng.choice(w)]]
        for _ in range(1, k):
            d = np.min(_sqdist(X, np.array(c)), axis=1) * w
            c.append(X[rng.choice(d)] if _ssum(d) > 0 else X[rng.choice(w)])
        starts.append(np.array(c))
    if init is not None:
        starts.append(np.array(init))
    best = None
    for c in starts:
        r = _lloyd(X, w, c, iters)
        if best is None or r[2] < best[2]:
            best = r
    return best[0], best[1]


class OnlineTeams:
    def __init__(self, p: OTParams | None = None):
        self.p = p or OTParams()
        self.rng = SplitMix64(self.p.seed)
        self.cam_sum, self.cam_n = {}, collections.Counter()
        self.samples = collections.defaultdict(list)    # track id -> [(frame, cam, sqrt hist)]
        self.dets = collections.defaultdict(list)       # track id -> [(frame, has_read)]
        self.centres = None                             # (k, D) in centred feature space
        self.roles = None                               # cluster -> "A" / "B" / "ref" / "other"
        self.radius = None
        self.prev_team = {}                             # "A"/"B" -> centre at the previous clustering
        self.log = []
        self.given = {}                                 # detection key -> label given (evaluation)

    # ---- colour ----
    def _centred(self, cam: str, s: np.ndarray) -> np.ndarray:
        g = sum(self.cam_sum.values()) / max(sum(self.cam_n.values()), 1)
        return s - self.cam_sum[cam] / self.cam_n[cam] + g

    def observe_crop(self, cam: str, hist: np.ndarray) -> np.ndarray:
        """Learn the camera mean from a good crop; returns its sqrt histogram."""
        s = np.sqrt(hist.astype(np.float32))
        self.cam_sum[cam] = self.cam_sum.get(cam, 0) + s
        self.cam_n[cam] += 1
        return s

    def feature(self, cam: str, s: np.ndarray) -> np.ndarray | None:
        return self._centred(cam, s) if self.cam_n[cam] else None

    # ---- labels ----
    def label(self, cam: str, s: np.ndarray) -> tuple[str | None, float]:
        if self.centres is None or not self.cam_n[cam]:
            return None, 0.0
        x = self._centred(cam, s)
        d = _sqdist(x[None], self.centres)[0]
        b = int(d.argmin())
        w = [math.exp(-(v - d[b]) / self.p.temperature) for v in d]
        tot = 0.0
        for v in w:
            tot += v
        w = [v / tot for v in w]
        if np.sqrt(d[b]) > self.radius[b]:
            return None, 0.0
        probs = collections.defaultdict(float)
        for j, r in enumerate(self.roles):
            probs[r] += w[j]
        r = self.roles[b]
        if r == UNKNOWN or probs[r] < self.p.label_p:
            return None, 0.0
        return r, probs[r]

    # ---- learning from the tracker ----
    def observe_track(self, tid: int, frame: int, cam: str | None, s: np.ndarray | None, has_read: bool) -> None:
        self.dets[tid].append((frame, has_read))
        if s is not None:
            self.samples[tid].append((frame, cam, s))

    def update(self, frame: int) -> None:
        p = self.p
        if frame < p.warmup or frame % p.every:
            return
        lo = frame - p.window
        protos, weights, rates = [], [], []
        for tid, ss in self.samples.items():
            if not ss or ss[-1][0] < lo:
                continue
            last = ss[-1][0]
            rec = [(c, s) for f, c, s in ss if f > max(lo, last - p.track_recent)]
            if len(rec) < p.min_samples:
                continue
            protos.append(_ssum(np.array([self._centred(c, s) for c, s in rec]), axis=0) / len(rec))
            weights.append(len(rec))
            ds = [r for f, r in self.dets[tid] if f > lo]
            rates.append((sum(ds), len(ds)))
        if len(protos) < max(p.min_tracks, p.k):
            return
        X, w = np.array(protos), np.array(weights, float)
        C, lab = weighted_kmeans(X, w, p.k, self.rng, self.centres,
                                 attempts=0 if p.warm_only and self.centres is not None else 10)
        share = np.array([_ssum(w[lab == j]) if (lab == j).any() else 0.0 for j in range(p.k)]) / _ssum(w)
        reads = np.array([sum(rates[i][0] for i in np.where(lab == j)[0]) for j in range(p.k)], float)
        dets = np.array([sum(rates[i][1] for i in np.where(lab == j)[0]) for j in range(p.k)], float)
        rate = reads / np.maximum(dets, 1)
        cand = [j for j in np.argsort(-rate, kind="stable") if share[j] >= p.min_team_share]
        if len(cand) < 2:
            return
        teams = cand[:2]
        rest = [j for j in range(p.k) if j not in teams]
        ref = min(rest, key=lambda j: rate[j]) if rest else None
        roles = [UNKNOWN] * p.k
        if ref is not None:
            roles[ref] = self.p.ref_label
        # team names follow the previous clustering's team centres
        if len(self.prev_team) == 2:
            a, b = self.prev_team["A"], self.prev_team["B"]
            d0 = _ssum((C[teams[0]] - a) ** 2) + _ssum((C[teams[1]] - b) ** 2)
            d1 = _ssum((C[teams[0]] - b) ** 2) + _ssum((C[teams[1]] - a) ** 2)
            if d1 < d0:
                teams = teams[::-1]
        roles[teams[0]], roles[teams[1]] = "A", "B"
        self.prev_team = {"A": C[teams[0]], "B": C[teams[1]]}
        # crop radius per cluster from the recent crops
        xs, cs = [], []
        for tid, ss in self.samples.items():
            for f, c, s in ss[-p.min_samples * 3:]:
                if f > lo:
                    xs.append(self._centred(c, s))
        xs = np.array(xs)
        dl = _sqdist(xs, C)
        nl = dl.argmin(1)
        dd = np.sqrt(dl[np.arange(len(xs)), nl])
        self.radius = np.array([np.percentile(dd[nl == j], p.radius_q) if (nl == j).any() else np.inf
                                for j in range(p.k)])
        self.centres, self.roles = C, roles
        self.log.append((frame, len(protos), [(roles[j], round(float(share[j]), 2), round(float(rate[j]), 4))
                                              for j in range(p.k)]))


def online_tracker_class(base, ot: OnlineTeams, hist_get, recorder=None, team_labels: bool = True):
    """WorldTracker subclass whose team labels and colour features come from `ot`: before each step the colour crops
    (hist_get(key) -> torso histogram or None) are labelled and learned from; after it, `ot` learns which court track
    each detection went to."""
    class Online(base):
        def step(self, frame, dets, info=None):
            if recorder is not None:
                recorder(self, frame, dets, info, hist_get)
            ot.update(frame)
            sq = {}
            for cam, meta in (info or {}).items():
                new = []
                for lab, h, lab_p, key in meta:
                    hh = hist_get(key)
                    lab, lab_p = None, 0.0
                    if hh is not None:
                        s = np.sqrt(hh.astype(np.float32))
                        if team_labels:
                            lab, lab_p = ot.label(cam, s)
                        ot.given[key] = lab
                        f = ot.feature(cam, s)
                        if f is not None:
                            self.feats[key] = f / np.float32(math.sqrt(_ssum(f.astype(np.float64) ** 2)) + 1e-9)
                        sq[key] = s
                        ot.observe_crop(cam, hh)
                    new.append((lab, h, lab_p, key))
                info[cam] = new
            super().step(frame, dets, info)
            for t in self.tracks:
                if t.last_hit != frame:
                    continue
                for k in reversed(t.dets):
                    if k[0] != frame:
                        break
                    ot.observe_track(t.tid, frame, k[1] if k in sq else None, sq.get(k), k in self.reads)
    return Online


def track_run_online(run, p=None, op: OTParams | None = None, stats: dict | None = None, recorder=None,
                     team_labels: bool = True, **track_kw):
    """track_run with team labels and colour features from OnlineTeams instead of the offline team model.
    recorder(tracker, frame, dets, info, hist_get): called with each frame's raw inputs (before labelling)."""
    from tools import appearance as ap
    from tools import world_tracker as wtm
    from tools.world_tracker import WTParams
    a = ap.load_appearance(run)
    sel = ap.good_crops(a)
    hist = {(int(f), str(c), str(g)): h for f, c, g, h in zip(a["frame"][sel], a["cam"][sel], a["gid"][sel],
                                                               a["hist"][sel])}
    ot = OnlineTeams(op)
    p = p or WTParams(online_relink=True)
    orig_cf, orig = wtm.colour_features, wtm.WorldTracker
    wtm.WorldTracker = online_tracker_class(orig, ot, hist.get, recorder, team_labels)
    wtm.colour_features = lambda r: {}
    try:
        track_kw.setdefault("appearance", "heights")
        T = wtm.track_run(run, p, **track_kw)
    finally:
        wtm.WorldTracker, wtm.colour_features = orig, orig_cf
    if stats is not None:
        stats["log"] = ot.log
        stats["given"] = ot.given
    return T


def read_inputs(path):
    """Frames of a CTIN2 input file (tools/courttrack_export, apps/courttrack probe): yields the camera table first
    ({index: (name, (x, y))}), then (frame, [(cam index, [detection dict])])."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    at = 0

    def take(fmt):
        nonlocal at
        v = struct.unpack_from(fmt, data, at)
        at += struct.calcsize(fmt)
        return v

    def text():
        n, = take("<i")
        nonlocal at
        s = data[at:at + n].decode()
        at += n
        return s

    assert data[:5] == b"CTIN2", "not a CTIN2 input file"
    at = 5
    n, = take("<i")
    cams = {}
    for i in range(n):
        x, y = take("<dd")
        cams[i] = (text(), (x, y))
    yield cams
    while True:
        frame, = take("<i")
        if frame < 0:
            return
        nc, = take("<i")
        out = []
        for _ in range(nc):
            cam, nd = take("<ii")
            dets = []
            for _ in range(nd):
                x, y, _tp, h = take("<dddd")
                gid, = take("<q")
                text()
                has_read, = take("<i")
                score, = take("<d")
                number = text()
                has_hist, = take("<i")
                hist = None
                if has_hist:
                    hist = np.frombuffer(data, "<f4", 144, at).copy()
                    at += 144 * 4
                dets.append(dict(x=x, y=y, h=h, gid=gid, read=(number, score) if has_read else None, hist=hist))
            out.append((cam, dets))
        yield frame, out


def replay_inputs(path, p=None, op: OTParams | None = None):
    """Python reference run on recorded tracker inputs (CTIN2): WorldTracker + OnlineTeams, as track_run_online.
    Returns (confirmed tracks, det_ident) with keys (frame, camera name, str(object id))."""
    from tools import world_tracker as wtm
    from tools.world_tracker import WTParams
    it = read_inputs(path)
    cams = next(it)
    hist = {}
    ot = OnlineTeams(op)
    wt = online_tracker_class(wtm.WorldTracker, ot, hist.get)(
        p or WTParams(online_relink=True), {}, set(), {name: np.array(xy) for name, xy in cams.values()})
    for frame, per_cam in it:
        hist.clear()
        dets, info = {}, {}
        for cam, ds in per_cam:
            name = cams[cam][0]
            seen = set()   # keep the first detection of each object id (as the probe and ct_replay)
            ds = [d for d in ds if not (d["gid"] in seen or seen.add(d["gid"]))]
            dets[name] = np.array([(d["x"], d["y"]) for d in ds]).reshape(-1, 2)
            info[name] = []
            for d in ds:
                key = (frame, name, str(d["gid"]))
                if d["read"] is not None:
                    wt.reads[key] = d["read"]
                    wt.attempts.add(key)
                if d["hist"] is not None:
                    hist[key] = d["hist"]
                info[name].append((None, d["h"], 0.0, key))
        wt.step(frame, dets, info)
    return wt.result(), wt.det_ident


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("runs", nargs="+")
    ap_.add_argument("--k", type=int, default=3)
    ap_.add_argument("--ref-label", default="other")
    a = ap_.parse_args()
    from tools import ROOT
    from tools.gt_eval import load_gt, score
    from tools.world_tracker import WTParams, identity_tracks, track_run
    for r in a.runs:
        run = ROOT / "experiments" / r
        gts = {n: load_gt(n, run) for n in ("game_2to3", "game_4to5")}
        st = {}
        T = track_run_online(run, op=OTParams(k=a.k, ref_label=a.ref_label), stats=st)
        G = identity_tracks(T, track_run.last_det_ident)
        on = [score(G, g)["idf1"] for g in gts.values()]
        T0 = track_run(run, WTParams(online_relink=True))
        G0 = identity_tracks(T0, track_run.last_det_ident)
        off = [score(G0, g)["idf1"] for g in gts.values()]
        first = st["log"][0][0] if st["log"] else None
        print(f"{r}: online {on[0]:.1%} {on[1]:.1%}   offline model {off[0]:.1%} {off[1]:.1%}   "
              f"first model at frame {first}; at 3600: {next((l for l in st['log'] if l[0] >= 3600), None)}",
              flush=True)


if __name__ == "__main__":
    main()
