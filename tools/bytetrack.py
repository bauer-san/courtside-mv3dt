"""ByteTrack per camera in place of MV3DT, feeding the court tracker (GitHub issue #12).

The question: how much of the result is MV3DT's doing? This replaces MV3DT with a plain 2D per-camera tracker and
keeps everything else of the default configuration, on the same detections.

  ByteTrack   Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022),
              re-implemented from the authors' yolox/tracker/byte_tracker.py (MIT): Kalman filter on (x, y, aspect,
              height), IoU matching of confident detections first (score fused into the IoU), then of the remaining
              tracks to the low-score detections, unconfirmed tracks, 30-frame lost buffer.
  input       detections.csv of a default run made with DET_LOG=1: every PeopleNet box the court filter kept (the
              detections MV3DT saw), per camera.
  foot point  the box's bottom centre projected onto the floor with the camera calibration (MV3DT uses a 3D body
              model instead, which also places people whose feet are hidden or cut off by the image edge).
  output      a run directory experiments/<run>_bt (or --suffix) in the format of a DeepStream run, so the court
              tracker, the ground truth and the harnesses score it unchanged:
                kafka.jsonl.gz   one message per camera and frame: ByteTrack ids, boxes, foot points
                jersey.npz       the source run's DeepStream jersey reads (made on MV3DT's boxes), each given to the
                                 ByteTrack box overlapping that box most (IoU >= 0.5)
                appearance.npz   torso colour histograms of the ByteTrack boxes (tools/appearance.py extract)
              camInfo_used, run.txt, team_model.npz and teams.yml are copied from the source run.

  python -m tools.bytetrack build bt_r1 [--track-thresh 0.5] [--suffix _bt]
"""

import argparse
import collections
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from tools import ROOT

FPS = 30


class KalmanFilter:
    """Constant-velocity Kalman filter on (cx, cy, aspect w/h, h), as in ByteTrack (from SORT/DeepSORT)."""
    std_pos, std_vel = 1 / 20, 1 / 160

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z):
        h = z[3]
        mean = np.r_[z, np.zeros(4)]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.H @ cov @ self.H.T + R
        c = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve(c, (cov @ self.H.T).T, check_finite=False).T
        return mean + K @ (z - self.H @ mean), cov - K @ S @ K.T


KF = KalmanFilter()
NEW, TRACKED, LOST, REMOVED = range(4)


class STrack:
    def __init__(self, tlwh, score):
        self.tlwh0 = np.asarray(tlwh, float)
        self.score = score
        self.mean = self.cov = None
        self.state = NEW
        self.activated = False
        self.tid = 0
        self.start = self.end_frame = 0

    @property
    def tlwh(self):
        if self.mean is None:
            return self.tlwh0.copy()
        cx, cy, a, h = self.mean[:4]
        return np.array([cx - a * h / 2, cy - h / 2, a * h, h])

    @staticmethod
    def xyah(tlwh):
        return np.array([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2] / tlwh[3], tlwh[3]])

    def activate(self, tid, frame):
        self.tid = tid
        self.mean, self.cov = KF.initiate(self.xyah(self.tlwh0))
        self.state, self.start, self.end_frame = TRACKED, frame, frame
        self.activated = frame == 1

    def update(self, det, frame):
        self.mean, self.cov = KF.update(self.mean, self.cov, self.xyah(det.tlwh0))
        self.state, self.activated, self.score, self.end_frame = TRACKED, True, det.score, frame


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of tlwh boxes, (len(a), len(b))."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = a.copy(), b.copy()
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def _dist(tracks, dets, fuse: bool) -> np.ndarray:
    iou = _iou(np.array([t.tlwh for t in tracks]).reshape(-1, 4), np.array([d.tlwh0 for d in dets]).reshape(-1, 4))
    if fuse and iou.size:
        iou = iou * np.array([d.score for d in dets])[None]
    return 1 - iou


def _assign(cost: np.ndarray, thresh: float):
    """Matches with cost <= thresh (lap.lapjv with cost_limit in the original), unmatched rows, unmatched cols."""
    if not cost.size:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    big = 1e5
    c = np.where(cost > thresh, big, cost)
    r, k = linear_sum_assignment(c)
    m = [(i, j) for i, j in zip(r, k) if c[i, j] < big]
    mr, mc = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class ByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, low_thresh=0.1, fuse_score=True):
        self.track_thresh, self.match_thresh, self.low = track_thresh, match_thresh, low_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_lost = int(FPS / 30 * track_buffer)
        self.fuse = fuse_score
        self.tracked, self.lost = [], []
        self.frame = 0
        self.next_id = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[STrack]:
        """boxes (N, 4) tlwh, scores (N,). Returns the confirmed, currently tracked tracks."""
        self.frame += 1
        f = self.frame
        hi = [STrack(b, s) for b, s in zip(boxes, scores) if s > self.track_thresh]
        lo = [STrack(b, s) for b, s in zip(boxes, scores) if self.low < s <= self.track_thresh]
        unconf = [t for t in self.tracked if not t.activated]
        confirmed = [t for t in self.tracked if t.activated]
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        for t in pool:
            if t.state != TRACKED:
                t.mean[7] = 0           # lost tracks keep their height
            t.mean, t.cov = KF.predict(t.mean, t.cov)
        activated, refound, lost, removed = [], [], [], []
        # 1. confident detections
        m, u_trk, u_det = _assign(_dist(pool, hi, self.fuse), self.match_thresh)
        for i, j in m:
            t = pool[i]
            (activated if t.state == TRACKED else refound).append(t)
            t.update(hi[j], f)
        # 2. low-score detections, tracks that were tracked in the previous frame
        rest = [pool[i] for i in u_trk if pool[i].state == TRACKED]
        m, u2, _ = _assign(_dist(rest, lo, False), 0.5)
        for i, j in m:
            rest[i].update(lo[j], f)
            activated.append(rest[i])
        for i in u2:
            rest[i].state = LOST
            lost.append(rest[i])
        # 3. unconfirmed tracks (one frame old) and the remaining confident detections
        hi = [hi[j] for j in u_det]
        m, u_unc, u_det = _assign(_dist(unconf, hi, self.fuse), 0.7)
        for i, j in m:
            unconf[i].update(hi[j], f)
            activated.append(unconf[i])
        for i in u_unc:
            unconf[i].state = REMOVED
            removed.append(unconf[i])
        # 4. new tracks
        for j in u_det:
            d = hi[j]
            if d.score >= self.det_thresh:
                self.next_id += 1
                d.activate(self.next_id, f)
                activated.append(d)
        for t in self.lost:
            if t.state == LOST and f - t.end_frame > self.max_lost:
                t.state = REMOVED
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked += [t for t in activated + refound if t not in self.tracked]
        self.lost = [t for t in self.lost + lost if t.state == LOST and t not in self.tracked]
        self._dedupe()
        return [t for t in self.tracked if t.activated]

    def _dedupe(self):
        """Tracked and lost tracks covering the same box (IoU > 0.85): keep the longer-lived one."""
        if not self.tracked or not self.lost:
            return
        d = _dist(self.tracked, self.lost, False)
        dup_a, dup_b = set(), set()
        for i, j in zip(*np.where(d < 0.15)):
            a, b = self.tracked[i], self.lost[j]
            if a.end_frame - a.start >= b.end_frame - b.start:
                dup_b.add(j)
            else:
                dup_a.add(i)
        self.tracked = [t for i, t in enumerate(self.tracked) if i not in dup_a]
        self.lost = [t for j, t in enumerate(self.lost) if j not in dup_b]


def load_detections_csv(path: Path) -> dict:
    """source id -> frame -> (boxes tlwh (N, 4), scores (N,))."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    out = collections.defaultdict(dict)
    order = np.lexsort((a[:, 1], a[:, 0]))
    a = a[order]
    keys = a[:, :2].astype(int)
    cut = np.flatnonzero(np.any(np.diff(keys, axis=0), axis=1)) + 1
    for blk in np.split(a, cut):
        out[int(blk[0, 1])][int(blk[0, 0])] = (blk[:, 2:6], blk[:, 6])
    return out


def source_cams(run: Path) -> dict:
    """Pipeline source id -> camera position, from the order of the dataset's video files (the configurator's
    source order; checked against the run's Kafka boxes by build())."""
    from tools.appearance import run_dataset
    from tools.stitch import SENSORS
    vids = sorted((run_dataset(run) / "videos").glob("*.mp4"))
    return {i: v.stem.split("_")[-1] for i, v in enumerate(vids)} if vids else dict(enumerate(SENSORS.values()))


def run_bytetrack(dets: dict, n_frames: int, **kw) -> dict:
    """source id -> frame -> [(track id, tlwh)]; every frame of the clip is a tracker step (empty if no boxes)."""
    out = {}
    for src, per in dets.items():
        bt = ByteTracker(**kw)
        res = {}
        for fr in range(n_frames):
            b, s = per.get(fr, (np.zeros((0, 4)), np.zeros(0)))
            res[fr] = [(t.tid, t.tlwh) for t in bt.update(b, s)]
        out[src] = res
    return out


def build(src: Path, dst: Path, appearance: bool = True, **kw) -> dict:
    from tools.appearance import extract, load_boxes
    from tools.stitch import SENSORS, floor_point, load_cameras
    dets = load_detections_csv(src / "detections.csv")
    cams = load_cameras(src / "camInfo_used")
    s2c = source_cams(src)
    sensor = {c: s for s, c in SENSORS.items()}
    n_frames = max(max(per) for per in dets.values()) + 1
    tracks = run_bytetrack(dets, n_frames, **kw)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("run.txt", "team_model.npz", "teams.yml"):
        shutil.copy(src / f, dst / f)
    shutil.copytree(src / "camInfo_used", dst / "camInfo_used", dirs_exist_ok=True)
    boxes = {}   # (cam, frame) -> [(id, l, t, r, b)]
    n_cut = n_out = 0
    with gzip.open(dst / "kafka.jsonl.gz", "wt") as f:
        for fr in range(n_frames):
            for s in sorted(tracks):
                cam = s2c[s]
                rows = tracks[s][fr]
                objs = []
                if rows:
                    tl = np.array([r[1] for r in rows])
                    u, v = tl[:, 0] + tl[:, 2] / 2, tl[:, 1] + tl[:, 3]
                    xy, ok = floor_point(cams[cam], u, v)
                    for (tid, b), (x, y), good, vv in zip(rows, xy, ok, v):
                        n_out += 1
                        n_cut += int(vv >= 1075)
                        if not good:
                            continue
                        l, t, w, h = (float(q) for q in b)
                        objs.append({"id": str(tid), "bbox": {"leftX": l, "topY": t, "rightX": l + w, "bottomY": t + h},
                                     "info": {"footLocation3D": f"{x:.4f},{y:.4f}"}})
                        boxes.setdefault((cam, fr), []).append((str(tid), l, t, l + w, t + h))
                f.write(json.dumps({"id": str(fr), "sensorId": sensor[cam], "objects": objs}) + "\n")
    # source id -> camera check: the run's own (MV3DT) boxes should overlap the ByteTrack boxes of the same camera
    src_boxes = load_boxes(src, 1)
    agree = collections.Counter()
    for (cam, fr), bs in list(boxes.items())[::50]:
        for c2 in cams:
            ref = src_boxes.get((c2, fr), [])
            if ref and bs:
                agree[cam, c2] += float((_iou(_tlwh(bs), _tlwh(ref)).max(axis=1) > 0.5).mean())
    for cam in cams:
        best = max(cams, key=lambda c2: agree[cam, c2])
        assert best == cam, f"source order: ByteTrack {cam} boxes match MV3DT {best} boxes best"
    stats = dict(frames=n_frames, boxes=n_out, cut_at_bottom=n_cut, tracks=sum(len({i for r in per.values()
                 for i, _ in r}) for per in tracks.values()))
    stats.update(_map_reads(src, dst, src_boxes, boxes))
    if appearance:
        extract(dst)
    (dst / "bytetrack.json").write_text(json.dumps(dict(source=src.name, params=kw, **stats), indent=1))
    return stats


def _tlwh(bs) -> np.ndarray:
    a = np.array([b[1:] for b in bs], float)
    a[:, 2:] -= a[:, :2]
    return a


def _map_reads(src: Path, dst: Path, src_boxes: dict, dst_boxes: dict) -> dict:
    """The source run's jersey reads, keyed to the ByteTrack box overlapping the read box most (IoU >= 0.5)."""
    from tools.jersey import load
    j = load(src)
    out = collections.defaultdict(list)
    lost = 0
    for fr, cam, gid, num, sc in zip(j["frame"], j["cam"], j["gid"], j["number"], j["score"]):
        ref = next((b for b in src_boxes.get((str(cam), int(fr)), []) if str(b[0]) == str(gid)), None)
        cand = dst_boxes.get((str(cam), int(fr)), [])
        if ref is None or not cand:
            lost += 1
            continue
        iou = _iou(_tlwh([ref]), _tlwh(cand))[0]
        k = int(iou.argmax())
        if iou[k] < 0.5:
            lost += 1
            continue
        key = (int(fr), str(cam), cand[k][0])
        out[key].append((str(num), float(sc)))
    keys = sorted(out)
    best = {k: max(out[k], key=lambda r: (r[0] != "", r[1])) for k in keys}   # two reads on one box: keep one
    np.savez_compressed(dst / "jersey.npz", frame=np.array([k[0] for k in keys]), cam=np.array([k[1] for k in keys]),
                        gid=np.array([k[2] for k in keys]), number=np.array([best[k][0] for k in keys]),
                        score=np.array([best[k][1] for k in keys], np.float32))
    return dict(read_attempts=len(j["frame"]), read_attempts_kept=len(j["frame"]) - lost,
                reads=int(sum(1 for k in keys if best[k][0])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("run", help="source run (made with DET_LOG=1)")
    ap.add_argument("--suffix", default="_bt")
    ap.add_argument("--track-thresh", type=float, default=0.5)
    ap.add_argument("--match-thresh", type=float, default=0.8)
    ap.add_argument("--track-buffer", type=int, default=30)
    ap.add_argument("--no-appearance", action="store_true")
    a = ap.parse_args()
    src = ROOT / "experiments" / a.run
    print(build(src, ROOT / "experiments" / (a.run + a.suffix), appearance=not a.no_appearance,
                track_thresh=a.track_thresh, match_thresh=a.match_thresh, track_buffer=a.track_buffer))


if __name__ == "__main__":
    main()
