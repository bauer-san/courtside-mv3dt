"""Court-plane multi-camera tracker over MV3DT's per-camera foot positions (option A2).

MV3DT's per-camera detections and 3D foot estimates are good, but its global IDs break whenever a player
leaves a camera's (narrow) view and the cross-camera hand-over fails, and it sometimes fuses different
players. This tracker ignores MV3DT's IDs and tracks people directly in court coordinates:

  input       per frame, every camera's detections (footLocation3D, metres) from kafka.jsonl.gz, after the
              court filter (bbox bottom back-projected onto the floor must be on court + margin)
  model       constant-velocity Kalman filter per person, state (x, y, vx, vy), dt = 1/30 s
  association per camera: Hungarian assignment of that camera's detections to predicted tracks
              (Mahalanobis gate); a track is updated with the mean of its assigned detections
              (at most one per camera, mutually consistent)
  births      unassigned detections from different cameras within `birth_radius` form one tentative track;
              confirmed after `confirm_hits` hits within `confirm_window` frames
  coasting    confirmed tracks survive `max_coast` frames without detections (view hand-overs, occlusions)
  merging     two confirmed tracks that stay within `merge_dist` for `merge_frames` frames while fed by
              *disjoint* camera sets are one person seen by two cameras: the younger is absorbed
  bias        per-camera constant offset (estimate_bias): least squares over mutual-nearest cross-camera
              detection pairs, offsets constrained to sum to zero
  appearance  optional (tools/appearance.py): team label per detection (torso colour; "other" = unknown) and
              body height (end-wall cameras). Association: penalty for a confidently labelled detection of
              the other team than a firmly established track team; tracks keep team votes and heights.
              Births and duplicate merges ignore team (per-camera colour differences would block true merges;
              the disjoint-camera rule already keeps two close players apart)
  numbers     optional (tools/jersey.py): jersey-number reads per detection. A track's number is established
              after num_min_reads reads agreeing >= num_share; identity = (team, number). Causal uses:
              association penalty for a confident read of another number; a track whose last
              num_switch_reads reads (within num_switch_window) all show the same other number is split there
              (identity swap); when a track first establishes its number it is re-linked to the most recent
              lost track with the same identity (gap <= id_gap); live tracks with the same identity merge;
              position-based re-linking/merging never joins conflicting established numbers
  relink      ONLINE (causal, deployable): when a track is confirmed, link it to the best recently lost
              confirmed track (same rules as post-stitching, using only the past); the lost track resumes
  main        track_default(): the causal configuration used in DeepStream (online team model, stored bias, no
              post); track_run() alone uses the offline inputs (whole-run bias, hand-named team model), kept as an
              upper reference
  post        OFFLINE (looks ahead; analysis only): drop static tracks near a court edge (seated spectators),
              then stitch track ends to later starts: jump <= d0 + vmax * gap; gap <= stitch_gap (3 s), or
              stitch_gap_team (8 s) when both tracks are confidently the same team; never across teams or
              heights differing > height_tol

  python -m tools.world_tracker <run> [--write] [--no-appearance]
"""

import argparse
import collections
import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools import ROOT
from tools.stitch import Params as StitchParams, ground_filter, load_cameras, load_detections

FPS = 30
UNKNOWN = "other"   # appearance label that carries no team information


@dataclass
class WTParams:
    meas_sigma: float = 0.35      # m, one camera's foot-position noise
    accel_sigma: float = 6.0      # m/s^2, player acceleration (process noise)
    gate_chi2: float = 13.8       # 2 dof, 99.9 %
    gate_base_m: float = 1.2      # gate radius for a track detected in the previous frame
    gate_speed: float = 5.0       # m/s: the gate widens by this while a track coasts ...
    gate_max_m: float = 3.0       # ... up to this radius
    consistency_m: float = 1.2    # detections assigned to one track must agree within this
    cost_logdet: bool = True      # cost = Mahalanobis + log det S (likelihood): uncertain (coasting) tracks no
                                  # longer look cheaper than well-tracked ones
    coast_vel_decay: float = 1.0  # per-frame velocity factor while a track coasts (1 = constant velocity)
    # direction-aware measurement noise: a camera's foot position is good across its line of sight and poor along
    # it (ground truth, tools/xcam.py: sd 0.26-0.28 m across, 0.45-0.75 m along); used in gating, fusion,
    # the consistency check, births and duplicate merging instead of the isotropic meas_sigma and fixed radii
    aniso: bool = True
    sigma_radial: float = 0.55
    sigma_tangential: float = 0.25
    consistency_chi2: float = 9.21  # 2 dof, 99 %: detections of one person from two cameras
    merge_chi2: float = 5.99        # 2 dof, 95 %: two tracks on one person (for merge_frames frames)
    # court edges (most track losses are within 1.5 m of a line): detections up to edge_margin beyond the lines may
    # continue a track but never start one; a track lost within edge_zone of a line coasts edge_coast frames
    edge_margin: float | None = None
    edge_zone: float = 1.0
    edge_coast: int | None = None
    # close encounters: when two tracks come within enc_dist, their identities (team, number) are noted; after
    # they separate beyond enc_sep, enc_window frames of colour labels and jersey reads decide whether they
    # swapped, and if so their histories before the encounter are exchanged (output revised enc_window late)
    encounters: bool = False
    enc_dist: float = 1.0
    enc_sep: float = 1.8
    enc_window: int = 45          # frames of evidence after separation
    enc_margin: float = 3.0       # swap only if the swapped assignment scores this much better
    enc_read_w: float = 3.0       # a jersey read counts as this many colour labels
    enc_max: int = 300            # frames: encounters that do not separate are dropped
    enc_need_read: bool = False   # swap only if a jersey read after separation fits the swapped identities
    # identity output: each detection is reported as the player (team:number) its track is identified as at that
    # frame; the roster is inferred online: a team:number enters it once roster_min_tracks tracks established it,
    # and a number alone identifies the player if only one roster team wears it
    roster_min_tracks: int = 2
    # per-player look (step 2): for a number both teams wear, the track's recent colour features are compared with
    # each candidate player's prototype (learned online from reliably identified tracks; the team's prototype
    # from its unique numbers until the player has one); the nearer one wins if closer by app_margin
    player_appearance: bool = True
    app_margin: float = 0.05      # Euclidean distance on unit-norm sqrt-histograms
    app_recent: int = 90          # frames of the track's features used
    app_min_samples: int = 3
    proto_min: int = 20           # samples before a prototype is used
    # identity freshness (loss breakdown: 38 % of wrong identities had recent reads of the true number; 46 % had no
    # read for 3 s): id_recent decides the identity's number from reads of the last id_window frames weighted by
    # exp(-age / id_tau) instead of all reads since the track began; id_doubt reports a track as unidentified after a
    # close encounter (another confirmed track within doubt_dist for doubt_frames) until a read supports it again
    id_recent: bool = True
    id_window: int = 300          # frames (10 s)
    id_tau: float = 150.0         # frames (weight halves every ~3.5 s)
    id_doubt: bool = False
    doubt_dist: float = 1.0       # m
    doubt_frames: int = 5
    birth_radius: float = 1.2     # m, merge unassigned detections of different cameras into one birth
    merge_dist: float = 0.9       # m
    merge_frames: int = 15
    confirm_hits: int = 8
    confirm_window: int = 12
    max_coast: int = 60           # frames (2 s) a confirmed track survives without detections
    tentative_coast: int = 3
    # appearance
    team_min_p: float = 0.8       # detection team label needs this probability (appearance.LABEL_P)
    team_penalty: float = 4.0     # added to the association cost for a detection of the other team ...
    team_penalty_p: float = 0.8   # ... if the detection label is at least this probable ...
    team_firm_votes: int = 10     # ... and the track team is firm (votes, share)
    team_firm_share: float = 0.85
    team_votes: float = 3.0       # a track's team is known after this many labelled detections ...
    team_share: float = 0.75      # ... agreeing at least this much
    # stitching
    stitch_gap: int = 90
    stitch_gap_team: int = 240
    stitch_d0: float = 1.0
    stitch_vmax: float = 5.0
    stitch_use_appearance: bool = True   # False: record labels/heights but never use them for stitching
    height_tol: float = 0.12      # m
    height_min_n: int = 5
    online_relink: bool = False
    # jersey numbers
    use_numbers: bool = True      # False: reads are recorded (for evaluation) but never used for tracking
    num_min_reads: int = 2
    num_share: float = 0.6
    num_penalty: float = 6.0      # association cost for a detection read as another number ...
    num_penalty_score: float = 0.8   # ... if the read is at least this confident
    num_switch_reads: int = 3
    num_switch_window: int = 90   # frames
    # swap split by majority: split when, within num_switch_window, >= num_switch_reads reads exist and another number
    # has >= split_share of them (the default rule needs num_switch_reads identical reads in a row, which one misread
    # blocks); the split is at that number's first read in the window
    split_majority: bool = True
    split_share: float = 0.66      # at least two thirds
    id_gap: int = 300             # frames (10 s): identity re-link window (60 s chained wrong identities)
    id_merge_dist: float = 2.0    # m: live tracks with the same identity closer than this merge
    # seated people (front-row spectators): MV3DT's standing-person model puts their feet ~2 m inside the court while
    # the box bottom, projected onto the floor, stays at their seat. Drop detections whose two foot points are more
    # than seated_gap apart while the box-bottom point is less than seated_edge inside the lines (None = off).
    # Removes 96 % of labelled spectator detections and 0.5 % of player detections (report section 4.9).
    seated_gap: float | None = 1.0
    seated_edge: float = 0.6
    num_needs_team: bool = True   # identity = team + number: number relinks/merges need both teams known and
                                  # equal, and a numbered track still pays the colour penalty (both teams
                                  # can wear the same number)
    # read scheduling (deployment budget): reading attempts per second over all players (None = every
    # available crop); tracks without a number first, identified tracks re-checked every reconfirm_s
    # identity logic (GitHub issue #11; held-out minute 5:00-6:00: +0.8 points identity IDF1, 6 of 7 runs better)
    team_recent: int | None = 90     # frames: the identity's team from the track's colour labels of this window
                                     # (team_votes labels, team_firm_share agreeing), else from all its labels
    read_low_score: float = 0.8      # in recent_number, a read below this recogniser score (74 % correct, the
    read_low_w: float = 0.25         # rest 96-100 %) counts this much
    split_close: bool = False        # after a close encounter (doubt_dist, doubt_frames) and before a read
                                     # supports the identity again, split_close_reads matching reads suffice
    split_close_reads: int = 2
    read_budget: float | None = None
    reconfirm_s: float = 3.0
    risk_dist: float | None = None   # m: if set, tracks this close are re-read unthrottled for 2 s after
                                     # (tested at 1.5 m: no gain, players are close almost constantly)


_UID = itertools.count()


@dataclass(eq=False)   # tracks compare by identity (ids can be exchanged between tracks)
class WTrack:
    tid: int
    x: np.ndarray                 # (4,) state
    P: np.ndarray                 # (4, 4) covariance
    born: int
    hits: list = field(default_factory=list)  # frames with detections
    last_hit: int = 0
    confirmed: bool = False
    out: list = field(default_factory=list)   # (frame, x, y, n_cams) for frames with detections
    cams: dict = field(default_factory=dict)  # frame -> frozenset of cameras that updated the track
    labels: list = field(default_factory=list)   # (frame, team) of labelled detections assigned to it
    heights: list = field(default_factory=list)  # (frame, metres)
    votes: collections.Counter = field(default_factory=collections.Counter)  # running tally of labels
    dets: list = field(default_factory=list)     # (frame, cam, gid) of every detection assigned to the track
    reads: list = field(default_factory=list)    # (frame, number, score) jersey reads of assigned detections
    nvotes: collections.Counter = field(default_factory=collections.Counter)
    last_read_try: int = -10 ** 9
    suspect: int = -10 ** 9        # frame of the last read that conflicted with the track's number
    end: str = ""                  # why it stopped: coast (no detections), split (number change), merged
    R: np.ndarray | None = None    # covariance of the last (fused) measurement
    id_linked: bool = False        # identity re-link already tried (once its identity became known)
    ident: str | None = None       # identity output: sticky, replaced only by a different established identity
    feats: list = field(default_factory=list)   # (frame, colour feature) of assigned detections (recent only)
    close_run: int = 0             # consecutive frames within doubt_dist of another confirmed track
    last_close: int = -10 ** 9     # last frame of a close encounter
    last_support: int = -10 ** 9   # last frame a read supported the track's identity
    uid: int = field(default_factory=lambda: next(_UID))   # never reused (id() of a freed track can be)

    def add_reads(self, reads: list) -> None:
        self.reads += reads
        self.nvotes.update(n for _, n, _ in reads)

    def number(self, p: WTParams) -> str | None:
        if not p.use_numbers:
            return None
        n = sum(self.nvotes.values())
        if n < p.num_min_reads:
            return None
        num, k = self.nvotes.most_common(1)[0]
        return num if k / n >= p.num_share else None

    def add_labels(self, labels: list) -> None:
        self.labels += labels
        self.votes.update(lab for _, lab in labels)

    def team(self, p: WTParams) -> str | None:
        c = self.votes
        n = sum(c.values())
        if n < p.team_votes:
            return None
        lab, k = c.most_common(1)[0]
        return lab if k / n >= p.team_share else None

    def firm_team(self, p: WTParams) -> str | None:
        c = self.votes
        n = sum(c.values())
        if n < p.team_firm_votes:
            return None
        lab, k = c.most_common(1)[0]
        return lab if k / n >= p.team_firm_share else None

    def height(self, p: WTParams) -> float | None:
        return float(np.median([h for _, h in self.heights])) if len(self.heights) >= p.height_min_n else None


def _matrices(p: WTParams):
    dt = 1.0 / FPS
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = G @ G.T * p.accel_sigma ** 2
    H = np.eye(2, 4)
    return F, Q, H


def _edge_dist(x: float, y: float) -> float:
    """Distance (m) from a floor point to the nearest court line (negative outside the court)."""
    from tools.evaluate import X_MAX, X_MIN, Y_HALF
    return min(x - X_MIN, X_MAX - x, Y_HALF - abs(y))


def _compatible(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b


def _conflict(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a != b


class WorldTracker:
    def __init__(self, p: WTParams | None = None, reads: dict | None = None, attempts: set | None = None,
                 cam_xy: dict | None = None):
        self.p = p or WTParams()
        self.cam_xy = cam_xy or {}  # camera -> floor position of the camera centre (for aniso)
        self.edge_keys: set = set()  # detection keys beyond the court margin: may only continue tracks
        self.reads = reads or {}    # (frame, cam, gid) -> (number, score)
        self.attempts = attempts if attempts is not None else set(self.reads)  # crops a reader could try
        self.tokens = self.p.read_budget or 0.0
        self.n_attempts = 0
        self.F, self.Q, self.H = _matrices(self.p)
        self.tracks: list[WTrack] = []
        self.finished: list[WTrack] = []
        self.next_id = 0
        self.close = collections.Counter()   # (tid_a, tid_b) -> consecutive frames close with disjoint cameras
        self.events: list = []               # (kind, frame, x, y): new / relink (at confirmation), lost, merge
        self.enc: dict = {}                  # (id(a), id(b)) -> encounter state
        self.enc_log: list | None = None     # set to [] to record encounter decisions
        self.roster = collections.defaultdict(collections.Counter)  # number -> team -> tracks that established it
        self._rostered: set = set()          # (track uid, team, number) already counted
        self.det_ident: dict = {}            # detection key -> identity at that frame (or None)
        self.feats: dict = {}                # detection key -> colour feature (player_appearance)
        self.proto: dict = {}                # identity -> [sum of features, n]
        self.team_proto: dict = {}           # team -> [sum of features, n] (from numbers only that team wears)

    def _gate(self, t: WTrack, z: np.ndarray, frame: int) -> float:
        S = self.H @ t.P @ self.H.T + np.eye(2) * self.p.meas_sigma ** 2
        d = z - t.x[:2]
        coast = max(0, frame - t.last_hit - 1) / FPS
        if np.linalg.norm(d) > min(self.p.gate_base_m + self.p.gate_speed * coast, self.p.gate_max_m):
            return np.inf
        m = float(d @ np.linalg.solve(S, d))
        return m if m <= self.p.gate_chi2 else np.inf

    def _ray(self, cam: str, Z: np.ndarray) -> np.ndarray | None:
        """Unit vectors from the camera to each floor point (N, 2), or None without aniso."""
        if not self.p.aniso or cam not in self.cam_xy:
            return None
        v = np.atleast_2d(Z) - self.cam_xy[cam]
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def _R(self, cam: str, Z: np.ndarray) -> np.ndarray:
        """Measurement covariances (N, 2, 2) of floor points Z seen by `cam`."""
        Z = np.atleast_2d(Z)
        u = self._ray(cam, Z)
        if u is None:
            return np.repeat(np.eye(2)[None] * self.p.meas_sigma ** 2, len(Z), axis=0)
        uu = np.einsum("ni,nj->nij", u, u)
        return self.p.sigma_radial ** 2 * uu + self.p.sigma_tangential ** 2 * (np.eye(2)[None] - uu)

    def _gate_matrix(self, Z: np.ndarray, frame: int, cam: str | None = None) -> np.ndarray:
        """Vectorised _gate for all tracks x detections (same result without aniso)."""
        X = np.array([t.x[:2] for t in self.tracks])
        R = self._R(cam, Z)                                                    # (N, 2, 2)
        S = np.array([t.P[:2, :2] for t in self.tracks])[:, None] + R[None]    # (T, N, 2, 2)
        D = Z[None, :, :] - X[:, None, :]
        m = np.einsum("tni,tnij,tnj->tn", D, np.linalg.inv(S), D)
        coast = np.array([max(0, frame - t.last_hit - 1) for t in self.tracks]) / FPS
        radius = np.minimum(self.p.gate_base_m + self.p.gate_speed * coast, self.p.gate_max_m)
        u = self._ray(cam, Z)
        if u is None:
            dist = np.linalg.norm(D, axis=2)
        else:   # the radius applies across the line of sight; along it, scaled by the noise ratio
            dr = np.einsum("tni,ni->tn", D, u)
            dt = np.einsum("tni,ni->tn", D, np.c_[-u[:, 1], u[:, 0]])
            dist = np.hypot(dr * self.p.sigma_tangential / self.p.sigma_radial, dt)
        ok = (dist <= radius[:, None]) & (m <= self.p.gate_chi2)
        if self.p.cost_logdet:
            m = m + np.log(np.linalg.det(S))
        return np.where(ok, m, np.inf)

    @staticmethod
    def _mahal(d: np.ndarray, S: np.ndarray) -> float:
        return float(d @ np.linalg.solve(S, d))

    def _fuse(self, cz: list) -> tuple[np.ndarray, np.ndarray]:
        """Fused floor point and covariance of detections [(cam, z, ...)]: inverse-covariance weighted."""
        if not self.p.aniso:
            zs = [c[1] for c in cz]
            return np.mean(zs, axis=0), np.eye(2) * self.p.meas_sigma ** 2 / len(zs)
        info, iz = np.zeros((2, 2)), np.zeros(2)
        for c in cz:
            Ri = np.linalg.inv(self._R(c[0], c[1])[0])
            info += Ri
            iz += Ri @ c[1]
        R = np.linalg.inv(info)
        return R @ iz, R

    def _near(self, g: list, cam: str, z: np.ndarray) -> bool:
        """Birth grouping: is detection z of `cam` the same person as group g (other cameras)?"""
        if not self.p.aniso:
            return np.linalg.norm(np.mean(g[1], axis=0) - z) < self.p.birth_radius
        zg, Rg = self._fuse(list(zip(g[4], g[1])))
        return self._mahal(z - zg, Rg + self._R(cam, z)[0]) <= self.p.consistency_chi2

    def step(self, frame: int, dets: dict[str, np.ndarray], info: dict[str, list] | None = None) -> None:
        """dets: camera -> (N, 2) foot positions; info: camera -> N x (team label or None, height, p, det key)."""
        p = self.p
        info = info or {}
        if p.read_budget is not None:
            self.tokens = min(p.read_budget, self.tokens + p.read_budget / FPS)
        for t in self.tracks:
            if p.coast_vel_decay != 1.0 and frame - t.last_hit > 1:
                t.x[2:] *= p.coast_vel_decay
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
        teams = [t.firm_team(p) for t in self.tracks]
        nums = [t.number(p) for t in self.tracks]
        assigned = collections.defaultdict(list)     # track index -> [(cam, z, label, height)]
        leftovers = []                                # (cam, z, label, height)
        for cam, Z in dets.items():
            if len(Z) == 0:
                continue
            meta = info.get(cam) or [(None, np.nan, 0.0, None)] * len(Z)
            if not self.tracks:
                leftovers += [(cam, z, m[0], m[1], m[3]) for z, m in zip(Z, meta)]
                continue
            C = self._gate_matrix(Z, frame, cam)
            for j, (lab, _, lab_p, _key) in enumerate(meta):
                if lab is not None and lab_p >= p.team_penalty_p:
                    for i, tt in enumerate(teams):
                        # a track identified by its number does not need (possibly wrong) colour labels
                        if tt is not None and tt != lab and (nums[i] is None or p.num_needs_team) \
                                and np.isfinite(C[i, j]):
                            C[i, j] += p.team_penalty
            for j, m in enumerate(meta):
                r = self.reads.get(m[3])
                if r and r[1] >= p.num_penalty_score:
                    for i, tn in enumerate(nums):
                        if tn is not None and tn != r[0] and np.isfinite(C[i, j]):
                            C[i, j] += p.num_penalty
            big = 1e9
            r, c = linear_sum_assignment(np.where(np.isfinite(C), C, big))
            used = set()
            for i, j in zip(r, c):
                if C[i, j] < big:
                    assigned[i].append((cam, Z[j], meta[j][0], meta[j][1], meta[j][3]))
                    used.add(j)
            leftovers += [(cam, Z[j], meta[j][0], meta[j][1], meta[j][3]) for j in range(len(Z)) if j not in used]
        # one track must not average two different people: keep detections consistent with the closest one
        for i, cz in list(assigned.items()):
            if len(cz) > 1 and p.aniso:
                pred = self.tracks[i].x[:2]
                cz.sort(key=lambda c: np.linalg.norm(c[1] - pred))
                ref, Rr = cz[0][1], self._R(cz[0][0], cz[0][1])[0]
                ok = [self._mahal(c[1] - ref, self._R(c[0], c[1])[0] + Rr) <= p.consistency_chi2 for c in cz]
                assigned[i] = [c for c, k in zip(cz, ok) if k]
                leftovers += [c for c, k in zip(cz, ok) if not k]
            elif len(cz) > 1:
                pred = self.tracks[i].x[:2]
                cz.sort(key=lambda c: np.linalg.norm(c[1] - pred))
                ref = cz[0][1]
                assigned[i] = [c for c in cz if np.linalg.norm(c[1] - ref) <= p.consistency_m]
                leftovers += [c for c in cz if np.linalg.norm(c[1] - ref) > p.consistency_m]
        # unidentified tracks first: they get the reading budget before identified ones are re-checked
        for i, cz in sorted(assigned.items(), key=lambda ic: nums[ic[0]] is not None):
            t = self.tracks[i]
            zs = [c[1] for c in cz]
            z, R = self._fuse(cz)
            t.R = R
            S = self.H @ t.P @ self.H.T + R
            K = t.P @ self.H.T @ np.linalg.inv(S)
            t.x = t.x + K @ (z - self.H @ t.x)
            t.P = (np.eye(4) - K @ self.H) @ t.P
            t.hits.append(frame)
            t.last_hit = frame
            t.out.append((frame, float(t.x[0]), float(t.x[1]), len(zs)))
            t.cams[frame] = frozenset(c[0] for c in cz)
            t.add_labels([(frame, c[2]) for c in cz if c[2] is not None])
            t.heights += [(frame, c[3]) for c in cz if np.isfinite(c[3])]
            t.dets += [c[4] for c in cz if c[4] is not None]
            prev_num = t.number(p)
            self._schedule_reads(t, frame, [c[4] for c in cz if c[4] is not None])
            if not t.confirmed and sum(h > frame - p.confirm_window for h in t.hits) >= p.confirm_hits:
                t.confirmed = True
                xy = np.mean([o[1:3] for o in t.out[:5]], axis=0)
                if p.online_relink:
                    t0, t = t, self._relink(t, frame)
                    self.events.append(("new" if t is t0 else "relink", frame, *xy))
                else:
                    self.events.append(("new", frame, *xy))
                prev_num = None      # a newly confirmed track with an established number re-links by identity
            if t.confirmed and self.reads and p.use_numbers:
                self._number_events(t, frame, prev_num)
        if p.read_budget is not None and p.risk_dist:
            self._mark_risk(frame)
        if self.edge_keys:
            leftovers = [c for c in leftovers if c[4] not in self.edge_keys]
        self._births(frame, leftovers)
        self._merge(frame)
        if p.encounters:
            self._encounters(frame)
        self._label_identities(frame)
        keep = []
        for t in self.tracks:
            limit = p.max_coast if t.confirmed else p.tentative_coast
            if t.confirmed and p.edge_coast and t.out and _edge_dist(t.out[-1][1], t.out[-1][2]) < p.edge_zone:
                limit = max(limit, p.edge_coast)
            if frame - t.last_hit <= limit:
                keep.append(t)
            else:
                t.end = t.end or "coast"
                if t.confirmed and t.out:
                    self.events.append(("lost", t.out[-1][0], *t.out[-1][1:3]))
                self.finished.append(t)
        self.tracks = keep

    def _relink(self, t: WTrack, frame: int) -> None:
        """Causal re-identification: attach a newly confirmed track to the best recently lost track."""
        p = self.p
        start_xy = np.mean([o[1:3] for o in t.out[:5]], axis=0)
        tt, th = t.team(p), t.height(p)
        best, best_cost = None, np.inf
        for lost in self.finished:
            if not lost.confirmed:
                continue
            lt, lh = lost.team(p), lost.height(p)
            if not _compatible(lt, tt) or (lh is not None and th is not None and abs(lh - th) > p.height_tol):
                continue
            if _conflict(lost.number(p), t.number(p)):
                continue
            same_team = lt is not None and lt == tt
            gap = t.out[0][0] - lost.out[-1][0]
            limit = p.stitch_gap_team if same_team else p.stitch_gap
            if gap < 1 or gap > limit:
                continue
            dist = float(np.linalg.norm(start_xy - np.mean([o[1:3] for o in lost.out[-5:]], axis=0)))
            allow = p.stitch_d0 + p.stitch_vmax * gap / FPS
            if dist > allow:
                continue
            cost = dist / allow + gap / limit + (0.0 if same_team else 0.3)
            if cost < best_cost:
                best, best_cost = lost, cost
        if best is None:
            return t
        best.out = best.out + t.out
        best.cams.update(t.cams)
        best.add_labels(t.labels)
        best.add_reads(t.reads)
        best.dets += t.dets
        best.heights += t.heights
        best.hits = best.hits + t.hits
        best.x, best.P, best.last_hit = t.x, t.P, t.last_hit
        self.finished.remove(best)
        self.tracks[self.tracks.index(t)] = best
        return best

    def _mark_risk(self, frame: int) -> None:
        """Swaps happen when players come close: such tracks are re-read without throttling for 2 s after."""
        live = [t for t in self.tracks if t.confirmed and t.last_hit == frame]
        if len(live) < 2:
            return
        X = np.array([t.x[:2] for t in live])
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        np.fill_diagonal(D, np.inf)
        for i in np.where((D < self.p.risk_dist).any(axis=1))[0]:
            live[i].suspect = frame

    def _schedule_reads(self, t: WTrack, frame: int, keys: list) -> None:
        p = self.p
        for k in keys:
            if k not in self.attempts:
                continue
            num = t.number(p)
            if (p.read_budget is not None and num is not None and frame - t.last_read_try < p.reconfirm_s * FPS
                    and frame - t.suspect > 2 * FPS):
                continue
            if p.read_budget is not None:
                if self.tokens < 1.0:
                    return
                self.tokens -= 1.0
            self.n_attempts += 1
            t.last_read_try = frame
            if k in self.reads:
                t.add_reads([(frame, *self.reads[k])])
                if num is not None and self.reads[k][0] != num:
                    t.suspect = frame          # possible swap: read this track again without waiting

    def _number_events(self, t: WTrack, frame: int, prev_num: str | None) -> None:
        p = self.p
        num = t.number(p)
        # re-link once the identity is complete: the number (and, with num_needs_team, a firm team, which often
        # comes later than the number)
        if not t.id_linked and num is not None and (not p.num_needs_team or t.firm_team(p) is not None):
            t.id_linked = True
            self._id_relink(t)
            return
        if prev_num is None:
            return
        if p.split_majority:
            win = [r for r in t.reads if frame - p.num_switch_window < r[0] <= frame]
            need = p.num_switch_reads
            if p.split_close and t.last_close > t.last_support:
                need = p.split_close_reads
            if len(win) >= need:
                n, k = collections.Counter(x for _, x, _ in win).most_common(1)[0]
                if n != prev_num and k / len(win) >= p.split_share:
                    self._split(t, min(g for g, x, _ in win if x == n))
            return
        recent = [r for r in t.reads if r[0] > frame - p.num_switch_window][-p.num_switch_reads:]
        if (len(recent) == p.num_switch_reads and len({n for _, n, _ in recent}) == 1
                and recent[0][1] != prev_num):
            self._split(t, recent[0][0])

    def _split(self, t: WTrack, since: int) -> None:
        """Identity swap detected: frames >= since belong to a new track (the old identity ends)."""
        new = WTrack(self.next_id, t.x.copy(), t.P.copy(), since, [h for h in t.hits if h >= since], t.last_hit,
                     True, [o for o in t.out if o[0] >= since], {f: c for f, c in t.cams.items() if f >= since})
        self.next_id += 1
        new.add_labels([lf for lf in t.labels if lf[0] >= since])
        new.add_reads([r for r in t.reads if r[0] >= since])
        new.heights = [hf for hf in t.heights if hf[0] >= since]
        new.dets = [k for k in t.dets if k[0] >= since]
        t.out = [o for o in t.out if o[0] < since]
        t.hits = [h for h in t.hits if h < since]
        t.cams = {f: c for f, c in t.cams.items() if f < since}
        t.labels = [lf for lf in t.labels if lf[0] < since]
        t.votes = collections.Counter(lab for _, lab in t.labels)
        t.reads = [r for r in t.reads if r[0] < since]
        t.nvotes = collections.Counter(n for _, n, _ in t.reads)
        t.heights = [hf for hf in t.heights if hf[0] < since]
        t.dets = [k for k in t.dets if k[0] < since]
        t.last_hit = t.out[-1][0] if t.out else since - 1
        self.tracks[self.tracks.index(t)] = new
        if t.out:
            t.end = "split"
            self.finished.append(t)
        self._id_relink(new)

    def _id_relink(self, t: WTrack) -> None:
        """Re-link a track that just established its jersey number to the last lost track of that identity."""
        p = self.p
        num, team = t.number(p), t.firm_team(p)
        best = None
        for lost in self.finished:
            if not lost.confirmed or lost.number(p) != num or not _compatible(lost.firm_team(p), team):
                continue
            if p.num_needs_team and (team is None or lost.firm_team(p) != team):
                continue
            gap = t.out[0][0] - lost.out[-1][0]
            if 1 <= gap <= p.id_gap and (best is None or lost.out[-1][0] > best.out[-1][0]):
                best = lost
        if best is None:
            return
        t.out = best.out + t.out
        t.cams = {**best.cams, **t.cams}
        t.labels = best.labels + t.labels
        t.votes = best.votes + t.votes
        t.reads = best.reads + t.reads
        t.nvotes = best.nvotes + t.nvotes
        t.heights = best.heights + t.heights
        t.dets = best.dets + t.dets
        t.hits = best.hits + t.hits
        t.born, t.tid = best.born, best.tid
        self.finished.remove(best)

    def roster_teams(self, num: str) -> set:
        return {tm for tm, k in self.roster[num].items() if k >= self.p.roster_min_tracks}

    def recent_number(self, t: WTrack, frame: int) -> str | None:
        """Number from the track's recent reads (age-weighted); same minimum reads and share as number()."""
        p = self.p
        w = collections.Counter()
        n = 0
        for g, x, s in reversed(t.reads):
            if g < frame - p.id_window:
                break
            if g <= frame:
                w[x] += np.exp(-(frame - g) / p.id_tau) * (p.read_low_w if s < p.read_low_score else 1.0)
                n += 1
        if n < p.num_min_reads:
            return None
        x, k = w.most_common(1)[0]
        tot = sum(w.values())
        return x if tot > 0 and k / tot >= p.num_share else None

    def identity(self, t: WTrack, frame: int | None = None) -> str | None:
        """The player a track is identified as now: team:number, or number alone if the roster makes it unique."""
        p = self.p
        num = self.recent_number(t, frame) if p.id_recent and frame is not None and p.use_numbers else t.number(p)
        if num is None or not t.confirmed:
            return None
        team = t.firm_team(p)
        if p.team_recent and frame is not None:
            c = collections.Counter(lab for g, lab in t.labels[-4 * p.team_recent:]
                                    if frame - p.team_recent < g <= frame)
            n = sum(c.values())
            if n >= p.team_votes:
                lab, k = c.most_common(1)[0]
                if k / n >= p.team_firm_share:
                    team = lab
        teams = self.roster_teams(num)
        if p.player_appearance and len(teams) > 1:
            by_look = self._closest_player(t, num, teams)
            if by_look is not None:
                return by_look
        if team is not None and team not in (UNKNOWN, "ref"):
            return f"{team}:{num}"
        if team is None and len(teams) == 1:
            return f"{next(iter(teams))}:{num}"
        return None

    def _proto(self, ident: str, team: str):
        for d, k in ((self.proto, ident), (self.team_proto, team)):
            v = d.get(k)
            if v is not None and v[1] >= self.p.proto_min:
                return v[0] / v[1]
        return None

    def _closest_player(self, t: WTrack, num: str, teams: set) -> str | None:
        p = self.p
        last = t.out[-1][0] if t.out else 0
        recent = [v for f, v in t.feats if f > last - p.app_recent]
        if len(recent) < p.app_min_samples:
            return None
        m = np.cumsum(np.array(recent), axis=0)[-1] / len(recent)     # summed in order (as the C++ port)
        d = []
        for tm in teams:
            pr = self._proto(f"{tm}:{num}", tm)
            if pr is None:
                return None
            # norm summed in order (not BLAS), so the C++ port (apps/courttrack) matches bit for bit
            d.append((float(np.sqrt(np.cumsum((m - pr).astype(np.float64) ** 2)[-1])), tm))
        d.sort()
        return f"{d[0][1]}:{num}" if d[1][0] - d[0][0] >= p.app_margin else None

    def _label_identities(self, frame: int) -> None:
        p = self.p
        if p.id_doubt or p.split_close:
            live = [t for t in self.tracks if t.confirmed and t.last_hit == frame]
            if len(live) > 1:
                X = np.array([t.x[:2] for t in live])
                D = np.linalg.norm(X[:, None] - X[None], axis=2)
                np.fill_diagonal(D, np.inf)
                near = D.min(axis=1) < p.doubt_dist
            else:
                near = np.zeros(len(live), bool)
            for t, c in zip(live, near):
                t.close_run = t.close_run + 1 if c else 0
                if t.close_run >= p.doubt_frames:
                    t.last_close = frame
        for t in self.tracks:
            if t.last_hit != frame:
                continue
            num, team = t.number(p), t.firm_team(p)
            if t.confirmed and num is not None and team is not None and team not in (UNKNOWN, "ref"):
                k = (t.uid, team, num)
                if k not in self._rostered:
                    self._rostered.add(k)
                    self.roster[num][team] += 1
            if p.player_appearance:
                fs = [(frame, self.feats[k]) for k in t.dets[-8:] if k[0] == frame and k in self.feats]
                if fs:
                    t.feats = [x for x in t.feats if x[0] > frame - p.app_recent] + fs
                    self._learn(t, num, team, [v for _, v in fs])
            new = self.identity(t, frame)
            if new is not None:
                if new != t.ident:
                    t.last_support = frame
                t.ident = new
            if t.ident is not None:
                num_shown = t.ident.split(":")[1]
                if any(g == frame and x == num_shown for g, x, _ in t.reads[-4:]):
                    t.last_support = frame
            shown = t.ident
            if p.id_doubt and shown is not None and t.last_close > t.last_support:
                shown = None
            for k in reversed(t.dets):
                if k[0] != frame:
                    break
                self.det_ident[k] = shown

    def _learn(self, t: WTrack, num: str | None, team: str | None, vs: list) -> None:
        """Prototype updates from reliable identities only: a number one roster team wears (team colour not
        contradicting), or a firm team colour that is one of the roster teams for the number."""
        if not t.confirmed or num is None:
            return
        teams = self.roster_teams(num)
        player_team = team if team not in (UNKNOWN, "ref") else None
        if len(teams) == 1:
            tm = next(iter(teams))
            if player_team not in (None, tm):
                return
            unique = True
        elif player_team is not None and player_team in teams:
            tm, unique = player_team, False
        else:
            return
        v = np.sum(vs, axis=0)
        for d, k in ((self.proto, f"{tm}:{num}"),) + (((self.team_proto, tm),) if unique else ()):
            if k in d:
                d[k][0] = d[k][0] + v
                d[k][1] += len(vs)
            else:
                d[k] = [v.copy(), len(vs)]

    def _identity_before(self, t: WTrack, s: int) -> tuple[str | None, str | None]:
        """(team, number) of a track from its evidence before frame s."""
        p = self.p
        c = collections.Counter(lab for f, lab in t.labels if f < s and lab != UNKNOWN)
        n = sum(c.values())
        team = None
        if n >= p.team_votes:
            lab, k = c.most_common(1)[0]
            team = lab if k / n >= p.team_share else None
        num = None
        if p.use_numbers:
            r = collections.Counter(x for f, x, _ in t.reads if f < s)
            m = sum(r.values())
            if m >= p.num_min_reads:
                x, k = r.most_common(1)[0]
                num = x if k / m >= p.num_share else None
        return team, num

    def _evidence(self, t: WTrack, f0: int, f1: int, ident: tuple) -> float:
        """How well a track's labels and reads in (f0, f1] fit an identity (agreements minus disagreements)."""
        team, num = ident
        sc = 0.0
        if team is not None:
            for f, lab in t.labels:
                if f0 < f <= f1 and lab != UNKNOWN:
                    sc += 1.0 if lab == team else -1.0
        if num is not None:
            for f, x, _ in t.reads:
                if f0 < f <= f1:
                    sc += self.p.enc_read_w * (1.0 if x == num else -1.0)
        return sc

    def _exchange_before(self, a: WTrack, b: WTrack, s: int) -> None:
        """Swap identities: a gets b's history before frame s (and its id), b gets a's."""
        def pre(t):
            return dict(out=[o for o in t.out if o[0] < s], hits=[h for h in t.hits if h < s],
                        cams={f: c for f, c in t.cams.items() if f < s}, labels=[x for x in t.labels if x[0] < s],
                        reads=[x for x in t.reads if x[0] < s], dets=[k for k in t.dets if k[0] < s],
                        heights=[x for x in t.heights if x[0] < s], tid=t.tid, born=t.born)
        pa, pb = pre(a), pre(b)
        for t, other in ((a, pb), (b, pa)):
            t.out = other["out"] + [o for o in t.out if o[0] >= s]
            t.hits = other["hits"] + [h for h in t.hits if h >= s]
            t.cams = {**other["cams"], **{f: c for f, c in t.cams.items() if f >= s}}
            t.labels = other["labels"] + [x for x in t.labels if x[0] >= s]
            t.reads = other["reads"] + [x for x in t.reads if x[0] >= s]
            t.dets = other["dets"] + [k for k in t.dets if k[0] >= s]
            t.heights = other["heights"] + [x for x in t.heights if x[0] >= s]
            t.votes = collections.Counter(lab for _, lab in t.labels)
            t.nvotes = collections.Counter(n for _, n, _ in t.reads)
            t.tid, t.born = other["tid"], other["born"]
        a.ident, b.ident = b.ident, a.ident

    def _encounters(self, frame: int) -> None:
        p = self.p
        live = [t for t in self.tracks if t.confirmed and frame - t.last_hit <= 1]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                if np.linalg.norm(a.x[:2] - b.x[:2]) < p.enc_dist:
                    key = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
                    e = self.enc.get(key)
                    if e is None:
                        self.enc[key] = dict(a=a, b=b, start=frame, sep=None,
                                             ia=self._identity_before(a, frame), ib=self._identity_before(b, frame))
                    else:
                        e["sep"] = None
        alive = {id(t) for t in self.tracks}
        for key, e in list(self.enc.items()):
            if key not in self.enc:     # dropped by an earlier swap in this loop
                continue
            a, b = e["a"], e["b"]
            if frame - e["start"] > p.enc_max:
                del self.enc[key]
                continue
            if e["sep"] is None:
                both = id(a) in alive and id(b) in alive
                if not both or np.linalg.norm(a.x[:2] - b.x[:2]) > p.enc_sep:
                    e["sep"] = frame
                continue
            if frame - e["sep"] < p.enc_window:
                continue
            f0, f1 = e["sep"], frame
            keep = self._evidence(a, f0, f1, e["ia"]) + self._evidence(b, f0, f1, e["ib"])
            swap = self._evidence(a, f0, f1, e["ib"]) + self._evidence(b, f0, f1, e["ia"])
            if self.enc_log is not None:   # diagnostics: the decision and the detections around it
                win = lambda t, lo, hi: [k for k in t.dets if lo <= k[0] <= hi]
                self.enc_log.append(dict(start=e["start"], sep=f0, frame=f1, keep=keep, swap=swap, ia=e["ia"],
                                         ib=e["ib"], a_pre=win(a, e["start"] - 120, e["start"]),
                                         b_pre=win(b, e["start"] - 120, e["start"]), a_post=win(a, f0, f1 + 60),
                                         b_post=win(b, f0, f1 + 60)))
            ok = swap - keep >= p.enc_margin
            if ok and p.enc_need_read:
                ok = any(f0 < f <= f1 and x == e["ib"][1] for f, x, _ in a.reads) or \
                    any(f0 < f <= f1 and x == e["ia"][1] for f, x, _ in b.reads)
            if ok:
                self._exchange_before(a, b, e["start"])
                self.events.append(("enc_swap", frame, *a.x[:2]))
                # pending encounters of these tracks refer to identities that just moved
                for k2, e2 in list(self.enc.items()):
                    if k2 != key and ({id(e2["a"]), id(e2["b"])} & {id(a), id(b)}):
                        del self.enc[k2]
            self.enc.pop(key, None)

    def _merge(self, frame: int) -> None:
        p = self.p
        live = [t for t in self.tracks if t.confirmed and t.last_hit == frame]
        if not live:        # no detections on this frame (detector interval): keep the closeness counters
            return
        seen = set()
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                ta, tb = live[a], live[b]
                key = (min(ta.tid, tb.tid), max(ta.tid, tb.tid))
                d = np.linalg.norm(ta.x[:2] - tb.x[:2])
                close = d < p.merge_dist
                if p.aniso and ta.R is not None and tb.R is not None:
                    close = self._mahal(ta.x[:2] - tb.x[:2], ta.R + tb.R) <= p.merge_chi2
                na, nb = ta.number(p), tb.number(p)
                if _conflict(na, nb):
                    continue
                same_id = na is not None and na == nb and _compatible(ta.team(p), tb.team(p))
                if p.num_needs_team:
                    same_id = same_id and ta.team(p) is not None and ta.team(p) == tb.team(p)
                if (close and not (ta.cams[frame] & tb.cams[frame])) or (same_id and d < p.id_merge_dist):
                    self.close[key] += 1
                    seen.add(key)
        for key in list(self.close):
            if key not in seen:
                del self.close[key]
        for (ia, ib), n in list(self.close.items()):
            if n < p.merge_frames:
                continue
            ta = next((t for t in self.tracks if t.tid == ia), None)
            tb = next((t for t in self.tracks if t.tid == ib), None)
            if ta is None or tb is None:
                continue
            old, new = (ta, tb) if ta.born <= tb.born else (tb, ta)
            # Only the close streak is known to be the same person: before it, the younger track may have
            # followed someone else, so its earlier history is dropped (mixing it in creates zigzags).
            since = frame - n + 1
            # Earlier frames of the younger track: where the older track was nearby they were the same person
            # (a duplicate from another camera) and are merged; elsewhere they are kept as a separate track.
            old_at = {o[0]: o for o in old.out}
            dup = {o[0] for o in new.out if o[0] < since and o[0] in old_at
                   and np.hypot(o[1] - old_at[o[0]][1], o[2] - old_at[o[0]][2]) < 1.5}
            since_set = {o[0] for o in new.out if o[0] >= since} | dup
            prefix = [o for o in new.out if o[0] not in since_set]
            if prefix:
                # keep the younger track's earlier history as its own (finished) track instead of losing it
                pre = WTrack(new.tid, new.x.copy(), new.P.copy(), new.born,
                             [h for h in new.hits if h not in since_set], prefix[-1][0], True, prefix,
                             {f: c for f, c in new.cams.items() if f not in since_set})
                pre.add_labels([lf for lf in new.labels if lf[0] not in since_set])
                pre.add_reads([r for r in new.reads if r[0] not in since_set])
                pre.dets = [k for k in new.dets if k[0] not in since_set]
                pre.heights = [hf for hf in new.heights if hf[0] not in since_set]
                pre.end = "merged"
                self.events.append(("merge", frame, *old.x[:2]))
                self.finished.append(pre)
            per = collections.defaultdict(list)
            for fr, x, y, k in old.out + [o for o in new.out if o[0] in since_set]:
                per[fr].append((x, y, k))
            old.out = [(fr, float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs])),
                        sum(v[2] for v in vs)) for fr, vs in sorted(per.items())]
            for fr, cs in new.cams.items():
                if fr in since_set:
                    old.cams[fr] = old.cams.get(fr, frozenset()) | cs
            old.add_labels([lf for lf in new.labels if lf[0] in since_set])
            old.add_reads([r for r in new.reads if r[0] in since_set])
            old.dets += [k for k in new.dets if k[0] in since_set]
            old.heights += [hf for hf in new.heights if hf[0] in since_set]
            old.hits = sorted(set(old.hits) | {h for h in new.hits if h in since_set})
            old.last_hit = max(old.last_hit, new.last_hit)
            old.x = (old.x + new.x) / 2
            old.P = (old.P + new.P) / 2
            self.tracks.remove(new)
            del self.close[(ia, ib)]

    def _births(self, frame: int, leftovers: list) -> None:
        groups = []                                   # [cams, [z], label, [(label, height, key)]]
        for cam, z, lab, h, key in sorted(leftovers, key=lambda c: c[0]):
            best = None
            for g in groups:
                if cam not in g[0] and self._near(g, cam, z):
                    best = g
                    break
            if best:
                best[0].add(cam)
                best[1].append(z)
                best[4].append(cam)
                best[2] = best[2] or lab
                best[3].append((lab, h, key))
            else:
                groups.append([{cam}, [z], lab, [(lab, h, key)], [cam]])
        for cams, zs, _, meta, gcams in groups:
            z, R = self._fuse([(c, zz) for c, zz in zip(gcams, zs)])
            P = np.diag([0.0, 0.0, 4.0, 4.0])
            P[:2, :2] = R
            t = WTrack(self.next_id, np.array([z[0], z[1], 0.0, 0.0]), P, frame, [frame], frame)
            t.R = R
            t.out.append((frame, float(z[0]), float(z[1]), len(zs)))
            t.cams[frame] = frozenset(cams)
            t.add_labels([(frame, lab) for lab, _, _ in meta if lab is not None])
            t.heights = [(frame, h) for _, h, _ in meta if np.isfinite(h)]
            t.dets = [k for _, _, k in meta if k is not None]
            if self.p.read_budget is None:
                t.add_reads([(frame, *self.reads[k]) for _, _, k in meta if k in self.reads])
            self.next_id += 1
            self.tracks.append(t)

    def result(self) -> list[WTrack]:
        return [t for t in self.finished + self.tracks if t.confirmed]


def _by_frame(d: dict, keep: np.ndarray, bias: dict | None = None, appearance: dict | None = None,
              p: WTParams | None = None):
    """frame -> cam -> [(x, y)] and frame -> cam -> [(team label or None, height)].
    MV3DT occasionally gives two people the same object id in one camera and frame; only the first is kept (as the
    DeepStream probe, apps/courttrack), since everything downstream is keyed by (frame, camera, id)."""
    p = p or WTParams()
    pos = collections.defaultdict(lambda: collections.defaultdict(list))
    info = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for fr, cam, gid, x, y in zip(d["frame"][keep], d["cam"][keep], d["gid"][keep], d["x"][keep], d["y"][keep]):
        if (int(fr), str(cam), str(gid)) in seen:
            continue
        seen.add((int(fr), str(cam), str(gid)))
        b = bias.get(cam, (0.0, 0.0)) if bias else (0.0, 0.0)
        pos[int(fr)][cam].append((x - b[0], y - b[1]))
        lab, h, lab_p = None, np.nan, 0.0
        if appearance:
            a = appearance.get((int(fr), str(cam), str(gid)))
            if a:
                probs, h = a
                best = max(probs, key=probs.get)
                if best != UNKNOWN and probs[best] >= p.team_min_p:
                    lab, lab_p = best, probs[best]
        info[int(fr)][cam].append((lab, h, lab_p, (int(fr), str(cam), str(gid))))
    return pos, info


def estimate_bias(by_frame, max_dist: float = 1.5) -> dict[str, tuple[float, float]]:
    """Per-camera constant offset from mutual-nearest cross-camera detection pairs (sum of offsets = 0)."""
    cams = sorted({c for v in by_frame.values() for c in v})
    idx = {c: i for i, c in enumerate(cams)}
    pair = collections.defaultdict(list)
    for per in by_frame.values():
        cs = sorted(per)
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                A, B = np.array(per[a]), np.array(per[b])
                D = np.linalg.norm(A[:, None] - B[None], axis=2)
                for k in range(len(A)):
                    j = D[k].argmin()
                    if D[:, j].argmin() == k and D[k, j] < max_dist:
                        pair[(a, b)].append(B[j] - A[k])
    rows, rhs, w = [], [], []
    for (a, b), v in pair.items():
        r = np.zeros(len(cams))
        r[idx[b]], r[idx[a]] = 1, -1          # bias_b - bias_a = median offset
        rows.append(r)
        rhs.append(np.median(v, axis=0))
        w.append(np.sqrt(len(v)))
    rows.append(np.ones(len(cams)))
    rhs.append(np.zeros(2))
    w.append(max(w, default=1.0))
    A = np.array(rows) * np.array(w)[:, None]
    B = np.array(rhs) * np.array(w)[:, None]
    sol = np.linalg.lstsq(A, B, rcond=None)[0]
    return {c: (float(sol[idx[c], 0]), float(sol[idx[c], 1])) for c in cams}


def seated(d: dict, cams: dict, gap: float, edge: float, frame_h: int = 1080) -> np.ndarray:
    """Detections of seated people: MV3DT foot point more than `gap` from the box-bottom floor point, and the box-bottom
    point less than `edge` inside the court lines. Boxes cut by the bottom image edge are never flagged."""
    from tools.stitch import floor_point
    out = np.zeros(len(d["frame"]), bool)
    for cam, P in cams.items():
        sel = np.where(d["cam"] == cam)[0]
        if not len(sel):
            continue
        xy, ok = floor_point(P, d["u"][sel], d["v"][sel])
        g = np.hypot(xy[:, 0] - d["x"][sel], xy[:, 1] - d["y"][sel])
        e = np.array([_edge_dist(x, y) for x, y in xy])
        out[sel] = ok & (g > gap) & (e < edge) & (d["v"][sel] < frame_h - 5)
    return out


def is_static_edge(t: WTrack, radius: float = 1.0, edge: float = 2.0) -> bool:
    """Track that never moves more than `radius` and sits within `edge` of a court line: a seated spectator."""
    from tools.evaluate import X_MAX, X_MIN, Y_HALF
    xy = np.array([o[1:3] for o in t.out])
    c = np.median(xy, axis=0)
    still = np.percentile(np.linalg.norm(xy - c, axis=1), 90) < radius
    return bool(still and min(c[0] - X_MIN, X_MAX - c[0], Y_HALF - abs(c[1])) < edge)


def post_stitch(tracks: list[WTrack], p: WTParams | None = None) -> list[WTrack]:
    """Greedy one-to-one linking of track ends to later starts, team- and height-aware."""
    p = p or WTParams()
    tracks = sorted(tracks, key=lambda t: t.out[0][0])
    starts = np.array([t.out[0][0] for t in tracks])
    info = [(t.team(p), t.height(p)) if p.stitch_use_appearance else (None, None) for t in tracks]
    cand = []
    for i, a in enumerate(tracks):
        end_f = a.out[-1][0]
        end_xy = np.mean([o[1:3] for o in a.out[-5:]], axis=0)
        lo = np.searchsorted(starts, end_f + 1)
        hi = np.searchsorted(starts, end_f + max(p.stitch_gap, p.stitch_gap_team), side="right")
        for j in range(lo, hi):
            b = tracks[j]
            (ta, ha), (tb, hb) = info[i], info[j]
            if not _compatible(ta, tb) or _conflict(a.number(p), b.number(p)):
                continue
            if ha is not None and hb is not None and abs(ha - hb) > p.height_tol:
                continue
            same_team = ta is not None and ta == tb
            gap = b.out[0][0] - end_f
            limit = p.stitch_gap_team if same_team else p.stitch_gap
            if gap > limit:
                continue
            dist = float(np.linalg.norm(np.mean([o[1:3] for o in b.out[:5]], axis=0) - end_xy))
            allow = p.stitch_d0 + p.stitch_vmax * gap / FPS
            if dist <= allow:
                cand.append((dist / allow + gap / limit + (0.0 if same_team else 0.3), i, j))
    cand.sort()
    succ, pred = {}, {}
    for _, i, j in cand:
        if i not in succ and j not in pred:
            succ[i], pred[j] = j, i
    out = []
    for i in range(len(tracks)):
        if i in pred:
            continue
        head = tracks[i]
        cur = i
        while cur in succ:
            cur = succ[cur]
            m = tracks[cur]
            head.out = head.out + m.out
            head.add_labels(m.labels)
            head.add_reads(m.reads)
            head.dets += m.dets
            head.heights += m.heights
        out.append(head)
    return out


def colour_features(run: Path) -> dict:
    """(frame, cam, gid) -> unit-norm, per-camera-centred sqrt torso histogram (good crops only)."""
    from tools.appearance import features, good_crops, load_appearance
    a = load_appearance(run)
    sel = good_crops(a)
    a = {k: v[sel] for k, v in a.items()}
    m = np.load(run / "team_model.npz")
    cam_mean = {str(c): v for c, v in zip(m["cams"], m["cam_means"])}
    cam_mean["_global"] = m["global_mean"]
    X = features(a, cam_mean)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return {(int(f), str(c), str(g)): x for f, c, g, x in zip(a["frame"], a["cam"], a["gid"], X)}


def detection_stride(pos: dict) -> int:
    """Frames between detector outputs (DeepStream detector interval; MV3DT with its visual tracker off outputs
    only on detector frames): the most common gap between frames that carry detections."""
    fr = np.array(sorted(pos))
    if len(fr) < 3:
        return 1
    gaps = np.diff(fr)
    return int(np.bincount(gaps[gaps <= 6]).argmax()) if (gaps <= 6).any() else 1


def for_stride(p: WTParams, stride: int) -> WTParams:
    """Rules that count detections per frame window, rescaled so they mean the same time span when detections come
    every `stride` frames (confirmation, tentative coasting, consecutive close detections before a merge)."""
    if stride <= 1:
        return p
    return replace(p, confirm_window=p.confirm_window * stride, tentative_coast=p.tentative_coast * stride,
                   merge_frames=max(1, round(p.merge_frames / stride)))


def track_run(run: Path, p: WTParams | None = None, sp: StitchParams | None = None, inside_fn=None,
              correct_bias: bool | str = True, post: bool = True, appearance: bool | str = True,
              numbers: bool = True, reads_file: str = "jersey.npz") -> list[WTrack]:
    from tools.evaluate import inside
    p = p or WTParams()
    inside_fn = inside_fn or inside
    sp = sp or StitchParams(margin=0.3)
    d = load_detections(run / "kafka.jsonl.gz")
    cams = load_cameras(run / "camInfo_used")
    keep = ground_filter(d, cams, inside_fn, sp)
    if p.seated_gap is not None:
        keep &= ~seated(d, cams, p.seated_gap, p.seated_edge, sp.frame_h)
    app = None
    if appearance == "heights" and (run / "appearance.npz").exists():
        # body heights of good crops only, no offline team model (track_run_online labels the crops itself)
        from tools.appearance import good_crops, load_appearance
        a = load_appearance(run)
        sel = good_crops(a)
        app = {(int(f), str(c), str(g)): ({UNKNOWN: 1.0}, float(h))
               for f, c, g, h in zip(a["frame"][sel], a["cam"][sel], a["gid"][sel], a["height"][sel])}
    elif appearance and (run / "appearance.npz").exists() and (run / "teams.yml").exists():
        from tools.appearance import team_probs
        app = team_probs(run)
    raw, _ = _by_frame(d, keep)
    if correct_bias == "stored":     # fixed values measured at installation (tools/calib_bias.py, in camInfo)
        from tools.calib_bias import stored
        bias = stored(run / "camInfo_used")
        if not bias:                 # runs recorded before the bias was stored: the current calibration's values
            import sys
            bias = stored()
            print(f"track_run: {run.name}/camInfo_used has no stored bias; using datasets/gym/camInfo",
                  file=sys.stderr)
        bias = bias or None
    else:                            # True: estimated over the whole run (offline; the reported results)
        bias = estimate_bias(raw) if correct_bias else None
    edge_keys = set()
    if p.edge_margin:
        wide = ground_filter(d, cams, inside_fn, StitchParams(margin=p.edge_margin))
        e = wide & ~keep
        edge_keys = {(int(f), str(c), str(g)) for f, c, g in zip(d["frame"][e], d["cam"][e], d["gid"][e])}
        keep = wide
    pos, info = _by_frame(d, keep, bias, app, p)
    p = for_stride(p, detection_stride(pos))
    reads, attempts = {}, set()
    if numbers and (run / reads_file).exists():
        from tools.jersey import load as load_reads
        j = load_reads(run, reads_file)
        for f, c, g, n, sc in zip(j["frame"], j["cam"], j["gid"], j["number"], j["score"]):
            k = (int(f), str(c), str(g))
            attempts.add(k)
            if n:
                reads[k] = (str(n), float(sc))
    cam_xy = {c: -np.linalg.solve(P[:, :3], P[:, 3])[:2] for c, P in cams.items()}
    wt = WorldTracker(p, reads, attempts, cam_xy)
    wt.edge_keys = edge_keys
    if p.player_appearance and (run / "appearance.npz").exists() and (run / "team_model.npz").exists():
        wt.feats = colour_features(run)
    wt.enc_log = [] if p.encounters else None
    for fr in range(min(pos), max(pos) + 1):
        wt.step(fr, {c: np.array(v) for c, v in pos.get(fr, {}).items()}, info.get(fr))
    tracks = wt.result()
    track_run.last_attempts = wt.n_attempts
    track_run.last_events = wt.events
    track_run.last_enc_log = wt.enc_log
    track_run.last_det_ident = wt.det_ident
    track_run.last_stride = detection_stride(pos)
    track_run.last_roster = {n: dict(c) for n, c in wt.roster.items()}
    if post:
        tracks = [t for t in tracks if not is_static_edge(t)]
        if not p.online_relink:
            tracks = post_stitch(tracks, p)
    return tracks


def track_default(run: Path, p: WTParams | None = None, op=None, team_labels: bool = True, **kw) -> list[WTrack]:
    """The main configuration, causal as deployed (the C++ port in DeepStream): online relinking, team labels and
    colour features from the online team model (tools/online_teams.py, 3 clusters), the fixed camera bias stored in
    camInfo at installation (tools/calib_bias.py), no offline spectator filter or post-stitching.
    team_labels=False keeps the online colour features but gives no team labels (ablation)."""
    from tools.online_teams import track_run_online
    kw.setdefault("correct_bias", "stored")
    kw.setdefault("post", False)
    return track_run_online(run, p or WTParams(online_relink=True), op, team_labels=team_labels, **kw)


@dataclass
class Identity:
    """Output unit with identity output: one player (team:number) or, before identification, one track."""
    label: str
    dets: list = field(default_factory=list)
    out: list = field(default_factory=list)     # (frame, x, y, n_cams)


def identity_tracks(tracks: list[WTrack], det_ident: dict) -> list[Identity]:
    """Regroup the tracks' detections and positions by the identity each detection had when it was made
    (causal); detections without an identity stay with their track."""
    groups = {}
    for t in tracks:
        cams_frame = collections.defaultdict(list)
        for k in t.dets:
            cams_frame[int(k[0])].append(k)
        pos = {o[0]: o for o in t.out}
        for f, keys in cams_frame.items():
            labs = [det_ident.get(k) for k in keys]
            lab = next((x for x in labs if x), None) or f"track{t.tid}"
            g = groups.setdefault(lab, Identity(lab))
            g.dets += keys
            if f in pos:
                g.out.append(pos[f])
    for g in groups.values():
        g.out.sort(key=lambda o: o[0])
    return list(groups.values())


def save(tracks: list[WTrack], path: Path) -> None:
    with path.open("w") as f:
        f.write("track,frame,x,y,n_cams\n")
        for i, t in enumerate(sorted(tracks, key=lambda t: t.out[0][0])):
            for fr, x, y, n in t.out:
                f.write(f"{i},{fr},{x:.3f},{y:.3f},{n}\n")


def main() -> None:
    from tools.evaluate import world_metrics
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-appearance", action="store_true")
    args = ap.parse_args()
    for r in args.runs:
        run = Path(r) if Path(r).is_dir() else ROOT / "experiments" / r
        tracks = track_run(run, appearance=not args.no_appearance)
        if args.write:
            save(tracks, run / "tracks_world.csv")
        m = world_metrics(tracks, run)
        print(run.name, " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()
