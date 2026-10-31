"""Camera-layout simulation: predicted court-tracker identity accuracy for candidate camera layouts.

The game's players move as they did in the recorded game: their court positions come from the live court tracker's
output of one run (courttrack.jsonl.gz, game frames only), its track pieces grouped by identity (labels near the
labelled moments, the live identities elsewhere) into one person per real player, plus referees; each has a body
height. For every camera of a layout (tools/camera_layout), every frame
and every person, the simulator produces what the court tracker would receive from DeepStream:

  detection     a box from the person's projected body (cylinder of the person's height); missed when mostly hidden
                behind nearer people, cut by the image border, too small, or suppressed by the detector's NMS
  foot point    MV3DT's fixed-height body model fitted to a noisy box: from the box bottom, or from the box top with a
                1.8 m model when the feet are hidden or outside the image; each camera's mean offset is removed, as the
                installation calibration (tools/calib_bias.py) does
  height        tools/appearance.person_height on the noisy box (height cameras only)
  jersey read   tried every 7th frame per person and camera on unhidden torsos of boxes >= 150 px; succeeds with a
                probability set by the box height and the body's image tilt (measured with upright crops), how squarely
                the camera faces the front or back of the player, and the player's own read factor for that minute
                (pose and the like, measured on the source run); some reads are wrong, at the player's own measured
                share and, mostly, as the values that player was actually misread as
  torso colour  every 3rd frame for boxes >= 80 px: a real torso histogram of the person's team, mixed with the
                occluding person's when the torso is partly hidden

These inputs go through the unchanged C++ court tracker and online team model (apps/courttrack/ct_replay), and the
identities it reports are scored against the simulated truth (tools/gt_eval.score) at the labelled moments of the real
game and over the whole game.

  python -m tools.layout_sim people [--run up_r1]                # build experiments/layout_sim/people.npz
  python -m tools.layout_sim stats --run up_r1 up_r2 up_r3       # measured statistics the models are fitted to
  python -m tools.layout_sim sim A --seeds 0 1 2 3 4 5 --tag val4 --keep        # layout A (validation)
  python -m tools.layout_sim sim C --height 5 --seeds 0 1 2 3 4 5 --tag screen  # a new layout at 5 m
  python -m tools.layout_sim summary --tags val4 screen --md reports/layout_sim.md
"""

import argparse
import collections
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import ROOT

OUT = ROOT / "experiments" / "layout_sim"
FEAT_DIM = 144          # torso HSV histogram, 18 hue x 8 saturation (apps/courttrack/courttrack.h)
TEAM_NAMES = ("red", "maroon", "ref")


# ---------------------------------------------------------------------------------------------------------------- people

def _tracks(run: Path, last: int) -> dict:
    """Court track id -> (frames, xy, team labels, identities, detections per frame) over frames <= last."""
    tr = collections.defaultdict(lambda: ([], [], [], [], []))
    with gzip.open(run / "courttrack.jsonl.gz", "rt") as f:
        for ln in f:
            m = json.loads(ln)
            if m["frame"] > last:
                break
            for p in m["players"]:
                t = tr[p["track"]]
                t[0].append(m["frame"])
                t[1].append((p["x"], p["y"]))
                t[2].append(p["team"])
                t[3].append(p["identity"])
                t[4].append([(d["sensor"], int(d["objectId"])) for d in p["detections"]])
    return {k: (np.array(v[0]), np.array(v[1]), v[2], v[3], v[4]) for k, v in tr.items()}


ROSTER = {"red": {"0", "1", "2", "3", "7"}, "maroon": {"0", "1", "3", "22", "24", "32", "94"}}
GT_WEIGHT = 10           # a labelled detection outweighs this many frames of the live tracker's identity
FIRM = 30                # identity votes needed before a track piece counts as known (joining rule)


def _team_names(T: dict) -> dict:
    """The live tracker's A/B -> red/maroon, by the roster numbers each label carries most."""
    votes = collections.Counter()
    for k in T:
        for idn in T[k][3]:
            if idn:
                t, num = idn.split(":")
                for name, nums in ROSTER.items():
                    votes[t, name] += num in nums and not all(num in r for r in ROSTER.values())
    a_red = votes["A", "red"] + votes["B", "maroon"] >= votes["A", "maroon"] + votes["B", "red"]
    return {"A": "red" if a_red else "maroon", "B": "maroon" if a_red else "red"}


def build_people(run: Path, max_gap: int = 150, vmax: float = 6.0, use_gt: bool = True) -> list:
    """People of the game from the court tracker's track pieces: one person per real player, built by identity.

    Each piece gets identity votes: the labelled identity of its detections near the labelled moments (gt_by_object,
    GT_WEIGHT each, if use_gt) and the live tracker's identity of each of its frames (1 each, A/B mapped to red/maroon);
    its identity is the top vote if that has >= FIRM votes and at least half of them, and is on the roster: identities
    with >= 5 labelled detections in the run's labelled minutes (early in the game the live tracker gives some players
    numbers they don't wear, before the team model settles). A player is the union of their
    pieces; where two overlap in time, the piece with more votes per frame wins the frame. Pieces without an identity
    but with a team label on >= 30 % of their frames fill a gap of a player of that team when they fit inside it and are
    reachable at vmax (+1 m) from both ends; those without a team label (>= 30 frames) are joined in time order when
    reachable (no overlap, gaps <= max_gap) into referees of >= 300 frames. Gaps up to
    max_gap frames are filled by linear interpolation; longer ones mean the player was off the court. Each person keeps,
    per frame, the track piece used (for people_check and the per-player read factors)."""
    from tools.gt_eval import game_end
    last = game_end(run)
    T = _tracks(run, last)
    to_name = _team_names(T)
    from tools.gt_eval import load_gt
    gt = gt_by_object(run) if use_gt else {}
    labelled = collections.Counter(v for m in ("game_2to3", "game_4to5", "game_5to6") for v in load_gt(m, run).values())
    roster = {i for i, n in labelled.items() if n >= 5}
    ident, strength, team_of = {}, {}, {}
    for k, (fr, _, tm, idn, dets) in T.items():
        v = collections.Counter()
        for f, i, ds in zip(fr, idn, dets):
            if i:
                v[f"{to_name[i[0]]}:{i.split(':')[1]}"] += 1
            for cam, gid in ds:
                g = gt.get((int(f), cam, gid))
                if g:
                    v[g] += GT_WEIGHT
        best = v.most_common(1)
        if best and best[0][1] >= FIRM and best[0][1] >= 0.5 * sum(v.values()) and best[0][0] in roster:
            ident[k] = best[0][0]
            strength[k] = best[0][1] / len(fr)
        firm = [to_name[x] for x in tm if x]
        team_of[k] = collections.Counter(firm).most_common(1)[0][0] if len(firm) >= 0.3 * len(fr) else None
    # players: frame -> (track, xy) by the strongest piece
    players = collections.defaultdict(dict)
    for k in sorted(ident, key=lambda k: strength[k]):          # stronger pieces overwrite weaker ones
        fr, xy = T[k][0], T[k][1]
        for f, p in zip(fr.tolist(), xy):
            players[ident[k]][f] = (k, p)
    # unidentified team pieces fill gaps
    def gap_fit(frames: dict, k) -> bool:
        fr, xy = T[k][0], T[k][1]
        f0, f1 = int(fr[0]), int(fr[-1])
        if any(f in frames for f in range(f0, f1 + 1)):
            return False
        before = [f for f in frames if f < f0]
        after = [f for f in frames if f > f1]
        ok = True
        if before:
            fb = max(before)
            ok &= np.hypot(*(xy[0] - frames[fb][1])) <= 1.0 + vmax * (f0 - fb) / 30
        if after:
            fa = min(after)
            ok &= np.hypot(*(xy[-1] - frames[fa][1])) <= 1.0 + vmax * (fa - f1) / 30
        return ok and (bool(before) or bool(after))
    loose = []
    for k in sorted((k for k in T if k not in ident), key=lambda k: -len(T[k][0])):
        tm = team_of[k]
        if tm is None:
            if len(T[k][0]) >= 30:
                loose.append(k)
            continue
        if len(T[k][0]) < 15:
            continue
        cands = [i for i in players if i.startswith(tm + ":") and gap_fit(players[i], k)]
        if cands:   # the player whose nearest end is closest in time
            def dist(i):
                fs = np.array(sorted(players[i]))
                return np.min(np.abs(fs - T[k][0][0])) + np.min(np.abs(fs - T[k][0][-1]))
            i = min(cands, key=dist)
            for f, p in zip(T[k][0].tolist(), T[k][1]):
                players[i][f] = (k, p)
    rng = np.random.default_rng(0)
    out = []
    def person(frames: dict, team: str, num: str, h: float) -> dict:
        fs = sorted(frames)
        fr, xy, used = [], [], []
        for a_, b_ in zip(fs, fs[1:] + [None]):
            fr.append(a_)
            xy.append(frames[a_][1])
            used.append(frames[a_][0])
            if b_ is not None and 1 < b_ - a_ <= max_gap:
                gap = np.arange(a_ + 1, b_)
                w = (gap - a_) / (b_ - a_)
                fr += gap.tolist()
                xy += list(frames[a_][1] + w[:, None] * (frames[b_][1] - frames[a_][1]))
                used += [-1] * len(gap)
        return dict(frames=np.array(fr), xy=np.array(xy), team=team, num=num, height=h, used=np.array(used),
                    tracks=sorted(set(used) - {-1}))
    for i in sorted(players):
        team, num = i.split(":")
        out.append(person(players[i], team, num, float(np.clip(rng.normal(1.88, 0.07), 1.70, 2.05))))
    # referees: pieces without a team joined in time order (reachable, no overlap), kept if >= 300 frames in all
    chains = []
    for k in sorted(loose, key=lambda k: T[k][0][0]):
        f2, p2 = int(T[k][0][0]), T[k][1][0]
        best, bd = None, None
        for ci, ch in enumerate(chains):
            fl = max(ch)
            d = np.hypot(*(p2 - ch[fl][1]))
            if fl < f2 <= fl + max_gap and d <= 1.0 + vmax * (f2 - fl) / 30 and (bd is None or d < bd):
                best, bd = ci, d
        frames = {f: (k, q) for f, q in zip(T[k][0].tolist(), T[k][1])}
        if best is None:
            chains.append(frames)
        else:
            chains[best].update(frames)
    for ch in chains:
        if len(ch) >= 300:
            out.append(person(ch, "ref", "", 1.80))
    return [p for p in out if len(p["frames"]) >= 90]


def people_check(people: list, run: Path) -> dict:
    """How well the people match the labels: each labelled detection (exact labelled moments) -> the court track
    holding it -> the person using that track in that frame; share with the right identity, a wrong one, or none."""
    from tools.gt_eval import game_end, load_gt
    T = _tracks(run, game_end(run))
    owner = {}
    for i, p in enumerate(people):
        for f, k in zip(p["frames"].tolist(), p["used"].tolist()):
            if k >= 0:
                owner[f, k] = i
    where = {}
    for k, (fr, _, _, _, dets) in T.items():
        for f, ds in zip(fr, dets):
            for cam, gid in ds:
                where[int(f), cam, gid] = k
    res = {}
    for m in ("game_2to3", "game_4to5", "game_5to6"):
        c = collections.Counter()
        for (f, cam, gid), ident in load_gt(m, run).items():
            k = where.get((int(f), cam, int(gid)))
            q = owner.get((int(f), k))
            if q is None:
                c["no person"] += 1
            else:
                c["right" if f"{people[q]['team']}:{people[q]['num']}" == ident else "wrong"] += 1
        res[m] = c
    return res


def save_people(people: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run = people[0].get("run", "") if people else ""
    np.savez_compressed(path, frames=np.concatenate([p["frames"] for p in people]),
                        xy=np.concatenate([p["xy"] for p in people]),
                        pid=np.concatenate([np.full(len(p["frames"]), i) for i, p in enumerate(people)]),
                        team=np.array([p["team"] for p in people]), num=np.array([p["num"] for p in people]),
                        height=np.array([p["height"] for p in people]),
                        used=np.concatenate([p["used"] for p in people]), run=np.array(run))


def load_people(path: Path = OUT / "people.npz") -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ measured statistics

def read_inputs(path: Path, last: int) -> tuple[list, dict]:
    """A recorded court-tracker input file (CTIN2, apps/courttrack/courttrack_probe.cpp): camera names, and
    (frame, camera, gid) -> (x, y, height, number or None, read score, histogram or None) for frames <= last."""
    import struct
    out = {}
    with open(path, "rb") as f:
        assert f.read(5) == b"CTIN2"
        n = struct.unpack("<i", f.read(4))[0]
        names = []
        for _ in range(n):
            f.read(16)
            names.append(f.read(struct.unpack("<i", f.read(4))[0]).decode())
        s = lambda: f.read(struct.unpack("<i", f.read(4))[0]).decode()
        while True:
            fr = struct.unpack("<i", f.read(4))[0]
            if fr < 0 or fr > last:
                break
            for _ in range(struct.unpack("<i", f.read(4))[0]):
                cam, nd = struct.unpack("<ii", f.read(8))
                for _ in range(nd):
                    x, y, _, h = struct.unpack("<dddd", f.read(32))
                    gid = struct.unpack("<q", f.read(8))[0]
                    s()
                    has, score = struct.unpack("<id", f.read(12))
                    num = s()
                    feat = None
                    if struct.unpack("<i", f.read(4))[0]:
                        feat = np.frombuffer(f.read(4 * FEAT_DIM), "<f4")
                    out[fr, names[cam], gid] = (x, y, h, num if has else None, score, feat)
    return names, out


def gt_by_object(run: Path, minutes=("game_2to3", "game_4to5", "game_5to6"), reach: int = 30) -> dict:
    """(frame, camera, gid) -> true identity: the labelled moments' detections, carried along the same MV3DT object
    up to `reach` frames before and after (MV3DT ids rarely change hands within a second)."""
    from tools.gt_eval import load_gt
    out = {}
    for m in minutes:
        for (fr, cam, gid), ident in load_gt(m, run).items():
            for d in range(-reach, reach + 1):
                out[fr + d, cam, int(gid)] = ident
    return out


READ_BIN = 25            # px bins of the read-rate curve
FAST = 1.5              # m/s: above it a player faces the way they run (read facing model)
TILT_BINS = (0, 5, 10, 15, 20, 30, 45, 180)   # deg, image tilt of the body axis
MINUTE = 1800            # frames: per-player read factors are per game minute
K_FACTOR = 5             # reads: shrinkage of the per-player-minute read factor towards 1
K_WRONG = 20             # reads: shrinkage of a player's wrong-read share towards the pooled share


def tilt_deg(P, x, y) -> float:
    """Image tilt (deg) of a standing body at floor point (x, y): the segment from 0.9 to 1.5 m above it, measured
    from the image vertical (as apps/court_filter/torso_objects.c computes it for upright crops)."""
    X = np.array([[x, y, 0.9, 1.0], [x, y, 1.5, 1.0]]) @ P.T
    d = X[1, :2] / X[1, 2] - X[0, :2] / X[0, 2]
    return abs(float(np.degrees(np.arctan2(d[0], -d[1]))))


def game_minute(frame: int) -> int:
    """Game minute (0 = 0:00-1:00 of the game clock, as the labelled minutes are named) of a frame of the time-aligned
    clips (NL's offset added back, as moments() takes it off)."""
    from tools.gt_eval import frame_offsets
    global _NL_OFF
    if _NL_OFF is None:
        _NL_OFF = frame_offsets(ROOT / "experiments" / "npct_r1").get("NL", 0)
    return (int(frame) + _NL_OFF) // MINUTE


_NL_OFF = None


def _tilt_bin(t: float) -> int:
    return int(np.searchsorted(TILT_BINS, t, side="right")) - 1


def look_down(C, x, y, z=1.3):
    """Angle (rad) at which a camera at C looks down at height z over floor point (x, y)."""
    return np.arctan2(C[2] - z, np.hypot(x - C[0], y - C[1]))


def measure(runs: list[Path]) -> dict:
    """Statistics of the real pipeline (game frames) that the simulator's models are fitted to:
      read model     digit reads per detection of uncut boxes >= 150 px by box height and image tilt of the body
                     (tilt_deg at the court tracker's foot point), pooled over the cameras; fitted as size curve x
                     tilt factor
      read facing    reads per detection of players running faster than FAST, by |cos| of the angle between their
                     running direction and the camera's line of sight: fits g = a + (1 - a) |cos|
      wrong reads    share of reads of labelled players that disagree with the label, kinds, and the read scores of
                     right and wrong reads
      colour         torso histograms of labelled players by true team and by player (>= 200 crops); referees: detections on court tracks without a
                     team label (>= 90 % of >= 300 frames)
      checks         detections per camera per frame, box heights, two cameras' foot points of one labelled person"""
    from tools.appearance import load_boxes
    from tools.camera_layout import centre
    from tools.gt_eval import game_end
    from tools.stitch import load_cameras
    st = dict(det_per_frame=collections.defaultdict(list), box_h=collections.defaultdict(list),
              read_st=collections.defaultdict(lambda: [0, 0]), facing=collections.defaultdict(lambda: [0, 0]),
              reads_ok=[], reads_bad=[], hist=collections.defaultdict(list), hist_id=collections.defaultdict(list),
              pair_d=collections.defaultdict(list))
    for run in runs:
        last = game_end(run)
        names, inp = read_inputs(run / "courttrack_inputs.bin", last)
        boxes = load_boxes(run, 1)
        Ps = load_cameras(run / "camInfo_used")
        C = {c: centre(P) for c, P in Ps.items()}
        gt = gt_by_object(run)
        frames = collections.Counter((cam, fr) for fr, cam, _ in inp)
        for cam in names:
            st["det_per_frame"][cam].append(sum(frames[cam, f] for f in range(1, last + 1)) / last)
        for (cam, fr), bs in boxes.items():
            for gid, l, t, r, b in bs:
                v = inp.get((fr, cam, int(gid)))
                if v is None:
                    continue
                st["box_h"][cam].append(b - t)
                if t <= 5 or b >= 1075 or l <= 2 or r >= 1918 or b - t < READ_MIN_PX:
                    continue
                c = st["read_st"][cam, int((b - t) // READ_BIN), _tilt_bin(tilt_deg(Ps[cam], v[0], v[1]))]
                c[1] += 1
                c[0] += v[3] is not None
        for k, v in inp.items():
            if v[3] is not None and k in gt:
                true = gt[k].split(":")[1]
                (st["reads_ok"] if v[3] == true else st["reads_bad"]).append((v[4], v[3], true, gt[k], game_minute(k[0])))
            if v[5] is not None and k in gt:
                st["hist"][gt[k].split(":")[0]].append(v[5])
                st["hist_id"][gt[k]].append(v[5])
        per = collections.defaultdict(dict)
        for k, v in inp.items():
            if k in gt:
                per[k[0], gt[k]][k[1]] = v[:2]
        for (fr, _), cams in per.items():
            cs = sorted(cams)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    st["pair_d"][cs[i] + "-" + cs[j]].append(np.hypot(*np.subtract(cams[cs[i]], cams[cs[j]])))
        # facing (running direction) and referee colour, from the live output's tracks
        nteam, ntot, dets_of = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        with gzip.open(run / "courttrack.jsonl.gz", "rt") as f:
            for ln in f:
                m = json.loads(ln)
                fr = m["frame"]
                if fr > last:
                    break
                for p in m["players"]:
                    ntot[p["track"]] += 1
                    nteam[p["track"]] += p["team"] is not None
                    sp = np.hypot(p["vx"], p["vy"])
                    for d in p["detections"]:
                        k = (fr, d["sensor"], int(d["objectId"]))
                        v = inp.get(k)
                        if v is None:
                            continue
                        dets_of[p["track"]].append(k)
                        if sp > FAST:
                            c = C[d["sensor"]]
                            u = np.array([p["x"] - c[0], p["y"] - c[1]])
                            cs = abs(p["vx"] * u[0] + p["vy"] * u[1]) / (sp * np.linalg.norm(u))
                            st["facing"][min(int(cs * 4), 3)][1] += 1
                            st["facing"][min(int(cs * 4), 3)][0] += v[3] is not None
        for t, n in ntot.items():
            if n >= 300 and nteam[t] <= 0.1 * n:
                st["hist"]["ref"] += [inp[k][5] for k in dets_of[t] if inp[k][5] is not None and k not in gt]
    return st


def fit(st: dict) -> dict:
    """Model parameters from measure(). Read model: rate(size, tilt) = curve[size bin] x tilt_factor[tilt bin]
    (tilt_factor[0] = 1), fitted to the pooled cells by alternating weighted means; size bins with < 200 detections
    read as nothing. Per player (labelled moments): reads and wrong reads, and the wrong values read."""
    nb = max(k for _, k, _ in st["read_st"]) + 1
    nt = len(TILT_BINS) - 1
    r, n = np.zeros((nb, nt)), np.zeros((nb, nt))
    for (_, k, t), (a_, b_) in st["read_st"].items():
        r[k, t] += a_
        n[k, t] += b_
    tf = np.ones(nt)
    for _ in range(50):
        curve = r.sum(1) / np.maximum((n * tf).sum(1), 1e-9)
        tf = r.sum(0) / np.maximum((n * curve[:, None]).sum(0), 1e-9)
        tf = tf / tf[0]
    curve = np.where(n.sum(1) >= 200, r.sum(1) / np.maximum((n * tf).sum(1), 1e-9), 0.0)
    f = [st["facing"][k][0] / max(1, st["facing"][k][1]) for k in range(4)]
    # g(c) = a + (1 - a) c through the sideways (c ~ 0.125) and head-on (c ~ 0.875) bins
    q = f[0] / f[3]
    a = (0.875 * q - 0.125) / ((1 - 0.125) - q * (1 - 0.875))
    ok, bad = st["reads_ok"], st["reads_bad"]
    kinds = collections.Counter("drop" if rr in t else "add" if t in rr else "other" for _, rr, t, _, _ in bad)
    per = collections.defaultdict(lambda: [0, []])        # per player, and per player and game minute
    for _, rr, _, ident, mi in ok:
        per[ident][0] += 1
        per[f"{ident}@{mi}"][0] += 1
    for _, rr, _, ident, mi in bad:
        for key in (ident, f"{ident}@{mi}"):
            per[key][0] += 1
            per[key][1].append(rr)
    return dict(read_curve=curve, read_bin=READ_BIN, tilt_factor=tf, read_cells_r=r, read_cells_n=n,
                facing_a=float(np.clip(a, 0, 1)), facing_bins=np.array(f),
                wrong=len(bad) / (len(ok) + len(bad)),
                wrong_kinds=np.array([kinds["drop"], kinds["add"], kinds["other"]], float) / max(1, len(bad)),
                score_ok=np.array([x[0] for x in ok]), score_bad=np.array([x[0] for x in bad]),
                **{f"reads_{i}": np.array([v[0], len(v[1])]) for i, v in per.items()},
                **{f"wrongvals_{i}": np.array(v[1] or [""]) for i, v in per.items()},
                **{f"hist_{t}": np.array(v, np.float32) for t, v in st["hist"].items()},
                **{f"hist_id_{t}": np.array(v, np.float32) for t, v in st["hist_id"].items() if len(v) >= 200})


def read_rate(m: dict, hpx: float, tilt: float) -> float:
    """Model read rate per detection (uncut box >= 150 px): size curve, held at its peak beyond it (larger numbers are
    assumed not to read worse; the few very large boxes measured are close to the far cameras), x tilt factor."""
    c = m["read_curve"]
    k = min(int(hpx // int(m["read_bin"])), len(c) - 1)
    return float(max(c[k], c[int(np.argmax(c))]) if k >= int(np.argmax(c)) else c[k]) * float(m["tilt_factor"][_tilt_bin(tilt)])


def player_factors(run: Path, people: dict, m: dict) -> dict:
    """Per player and game minute: observed reads / reads predicted by read_rate, on the run the people were built
    from (each read-eligible detection -> its court track -> the person using that track in that frame), shrunk towards
    1 by K_FACTOR reads. Covers what the geometry doesn't: pose (leaning, bending, turning away), free throws facing a
    camera, individual shirts."""
    from tools.appearance import load_boxes
    from tools.gt_eval import game_end
    from tools.stitch import load_cameras
    last = game_end(run)
    T = _tracks(run, last)
    _, inp = read_inputs(run / "courttrack_inputs.bin", last)
    Ps = load_cameras(run / "camInfo_used")
    where = {}
    for k, (fr, _, _, _, dets) in T.items():
        for f, ds in zip(fr, dets):
            for cam, gid in ds:
                where[int(f), cam, gid] = k
    owner = {}
    for f, q, k in zip(people["frames"].tolist(), people["pid"].tolist(), people["used"].tolist()):
        if k >= 0:
            owner[f, k] = q
    obs, exp = collections.Counter(), collections.Counter()
    for (cam, fr), bs in load_boxes(run, 1).items():
        if fr > last:
            continue
        for gid, l, t, r, b in bs:
            v = inp.get((fr, cam, int(gid)))
            if v is None or t <= 5 or b >= 1075 or l <= 2 or r >= 1918 or b - t < READ_MIN_PX:
                continue
            q = owner.get((fr, where.get((fr, cam, int(gid)))))
            if q is None or people["team"][q] == "ref":
                continue
            key = (f"{people['team'][q]}:{people['num'][q]}", game_minute(fr))
            exp[key] += read_rate(m, b - t, tilt_deg(Ps[cam], v[0], v[1]))
            obs[key] += v[3] is not None
    out = collections.defaultdict(lambda: np.ones(game_minute(last) + 1))
    for (ident, mi), e in exp.items():
        out[ident][mi] = (obs[ident, mi] + K_FACTOR) / (e + K_FACTOR)
    return dict(out)


def summarise(st: dict, m: dict) -> str:
    L = ["detections per camera per game frame: " + ", ".join(f"{c} {np.mean(v):.2f}" for c, v in
                                                             sorted(st["det_per_frame"].items()))]
    L.append("box height (px) of detections, 5/50/95 %: " + ", ".join(
        f"{c} {np.percentile(v, 5):.0f}/{np.median(v):.0f}/{np.percentile(v, 95):.0f}" for c, v in sorted(st["box_h"].items())))
    L.append("digit reads per detection, uncut boxes >= 150 px: size curve (tilt 0-5 deg) and tilt factor")
    for k, v in enumerate(m["read_curve"]):
        if m["read_cells_n"][k].sum() >= 200:
            L.append(f"  {k * READ_BIN:3d}-{k * READ_BIN + READ_BIN - 1:3d} px: {v:5.2%} (n={int(m['read_cells_n'][k].sum())})")
    L.append("  tilt factor " + ", ".join(f"{TILT_BINS[i]}-{TILT_BINS[i + 1]} deg {f:.2f}" for i, f in enumerate(m["tilt_factor"])))
    r, n = m["read_cells_r"], m["read_cells_n"]
    pred = m["read_curve"][:, None] * m["tilt_factor"][None, :] * n
    sel = n >= 200
    L.append(f"  fit: reads predicted / observed over cells with >= 200 detections {pred[sel].sum() / r[sel].sum():.3f}; "
             f"correlation of cell rates {np.corrcoef(pred[sel] / n[sel], r[sel] / n[sel])[0, 1]:.2f} ({sel.sum()} cells)")
    L.append("reads per detection of players running > %.1f m/s by |cos(running direction, line of sight)| "
             "0-0.25/../0.75-1: %s; fitted a = %.2f" % (FAST, " ".join(f"{x:.2%}" for x in m["facing_bins"]), m["facing_a"]))
    L.append(f"wrong reads of labelled players: {m['wrong']:.1%} (digit dropped / added / other: "
             + " / ".join(f"{x:.0%}" for x in m["wrong_kinds"]) + f"); median score right {np.median(m['score_ok']):.2f}, "
             f"wrong {np.median(m['score_bad']):.2f}")
    L.append("torso histograms: " + ", ".join(f"{t} {len(v)}" for t, v in sorted(st["hist"].items())))
    L.append("same person, same frame, two cameras: median (90 %) distance " + ", ".join(
        f"{p} {np.median(v):.2f} ({np.percentile(v, 90):.2f}) m" for p, v in sorted(st["pair_d"].items())))
    return "\n".join(L)


# --------------------------------------------------------------------------------------------------------- simulation

W, H = 1920, 1080
BODY_R = 0.25           # m, radius of the body cylinder
MODEL_H, MODEL_R = 1.8, 0.3   # MV3DT's person model (camInfo modelInfo)
REINFER = 7             # a person's torso is read every 7th frame (secondary-reinfer-interval 6)
COLOUR_EVERY, MIN_BOX_H = 3, 80   # tools/courttrack_config: torso colour every 3rd frame, boxes >= 80 px
READ_MIN_PX = 150       # torso objects for the jersey reader: boxes >= 150 px, not cut by the image border
COURT_MARGIN = 0.3      # court tracker's court filter (tools/courttrack_config)
HEIGHT_MAX_LOOK_DOWN = 20.0   # deg: new layouts give body heights from views at most this steep (FL/FR: 9-15 deg)


@dataclass
class SimParams:
    box_noise: float = 0.02       # box edge noise, share of the box height (1 sd); with los_sd fitted to the
                                  # measured cross-camera distances of one person (layout A, upright runs)
    noise_rho: float = 0.9        # frame-to-frame correlation of the box noise
    min_visible: float = 0.4      # share of the body inside the image and not hidden that the detector needs
    min_in_image: float = 0.5     # share of the body height inside the image
    min_px: float = 30.0          # smallest detectable box height
    miss: float = 0.02            # other misses
    nms_iou: float = 0.5          # detector NMS between full boxes
    read_scale: float = 0.9       # scales the measured read curve (fitted so layout A gives the measured reads)
    read_hidden: float = 0.3      # no read when more of the torso is hidden
    colour_mix: float = 0.25      # torso hidden beyond this share: colour mixed with the occluder's
    cal_err: float = 0.0          # calibration error: smooth per-camera position error field, m (1 sd per axis) at
                                  # 15 m from the camera, proportional to the distance
    lean_sd: float = 0.0          # body lean: horizontal offset of the top of the body, m (1 sd per axis), the same
                                  # for every camera, so MV3DT's upright body model misplaces the feet along each
                                  # camera's line of sight (opposite ends disagree, cameras on one mount agree)
    lean_rho: float = 0.97        # frame-to-frame correlation of the lean (about 1 s)
    los_sd: float = 0.25          # depth error along the line of sight, m (1 sd) at 10 m from the camera and
                                  # proportional to the distance, per person and camera position:
                                  # shared by cameras on one mount (they see the person along the same line), opposite
                                  # for cameras at opposite ends (measured: those disagree most, same-mount pairs least)
    los_rho: float = 0.97         # its frame-to-frame correlation


def layout_cameras(letter: str, zw: float | None) -> tuple[dict, set, set]:
    """Projection matrices of a layout (tools/camera_layout, lenses and aims chosen there with the game's occupancy),
    the colour cameras and the height cameras (None: by look-down angle). Cached in experiments/layout_sim/cams."""
    from tools import camera_layout as cl
    if letter == "A":
        cams = cl.current_layout()
        return cams, {"FL", "FR", "NR"}, {"FL", "FR"}
    f = OUT / "cams" / f"{letter}_{zw:.1f}.npz"
    if not f.exists():
        global _OCC
        if "_OCC" not in globals():
            _OCC = cl.occupancy(ROOT / "experiments" / "js_r1")
        (cams, _), = cl.layouts(zw, _OCC, only=letter).values()
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **cams)
    z = np.load(f)
    cams = {k: z[k] for k in z.files}
    return cams, set(cams), None


def _centre(P):
    return -np.linalg.solve(P[:, :3], P[:, 3])


def _plane_point(P, u, v, z0):
    """Back-project pixel (u, v) onto the plane at height z0."""
    Hm = np.c_[P[:, 0], P[:, 1], P[:, 2] * z0 + P[:, 3]]
    X = np.linalg.solve(Hm, np.array([u, v, 1.0]))
    return X[:2] / X[2]


def _boxes(P, x, y, h, radius=BODY_R, lean=None):
    """Image boxes (n, 4: l, t, r, b) and projective depth (> 0 in front of the camera; the calibrated matrices are
    scaled, so it orders people within one camera but isn't metres) of standing cylinders (radius BODY_R, height h),
    their top ring shifted horizontally by lean (n, 2; m) when given."""
    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    n = len(x)
    X = np.empty((n, 16, 4))
    X[:, :8, 0] = X[:, 8:, 0] = x[:, None] + radius * np.cos(a)
    X[:, :8, 1] = X[:, 8:, 1] = y[:, None] + radius * np.sin(a)
    X[:, :8, 2] = 0
    X[:, 8:, 2] = h[:, None]
    if lean is not None:
        X[:, 8:, 0] += lean[:, :1]
        X[:, 8:, 1] += lean[:, 1:]
    X[:, :, 3] = 1
    p = X @ P.T
    dep = p[..., 2]
    uv = p[..., :2] / np.where(dep > 0, dep, np.nan)[..., None]
    box = np.stack([uv[..., 0].min(1), uv[..., 1].min(1), uv[..., 0].max(1), uv[..., 1].max(1)], 1)
    return box, dep.min(1)


def _model_foot(P, p, box, feet_hidden):
    """MV3DT's foot point for an observed box: the floor point where its person model (MODEL_H tall, MODEL_R wide)
    projects to the box's horizontal centre and bottom edge (its top edge when the feet are hidden or below the image),
    solved by one Newton step from the true position p (the residual is small: box noise and the person's height)."""
    l, t, r, b = box
    eps = 0.05
    pts = np.array([p, p + (eps, 0), p + (0, eps)])
    mb, _ = _boxes(P, pts[:, 0], pts[:, 1], np.full(3, MODEL_H), MODEL_R)
    k = 1 if feet_hidden else 3
    f = np.c_[(mb[:, 0] + mb[:, 2]) / 2, mb[:, k]]
    J = (f[1:] - f[0]).T / eps
    res = np.array([(l + r) / 2, t if feet_hidden else b]) - f[0]
    try:
        return p + np.linalg.solve(J, res)
    except np.linalg.LinAlgError:
        return p.copy()


def _person_height(P, l, t, r, b):
    """tools/appearance.person_height (apps/courttrack/courttrack_probe.cpp)."""
    if t <= 5 or b >= H - 5 or l <= 2 or r >= W - 2:
        return np.nan
    x, y = _plane_point(P, (l + r) / 2, b, 0.0)
    hh = 0.5 + np.arange(211) * (2.1 / 210)
    v = (P[1, 0] * x + P[1, 1] * y + P[1, 2] * hh + P[1, 3]) / (P[2, 0] * x + P[2, 1] * y + P[2, 2] * hh + P[2, 3])
    i = int(np.argmin(np.abs(v - t)))
    return 0.5 + i * (2.1 / 210) if 0 < i < 210 else np.nan


def _covered(box, near, rows):
    """Share of a box's sample points (central 60 % of its width, rows given as shares of its height, inside the image)
    covered by the silhouettes (central 60 % of the width) of nearer boxes."""
    l, t, r, b = box
    us = l + (r - l) * np.linspace(0.2, 0.8, 5)
    vs = t + (b - t) * np.asarray(rows)
    U, V = np.meshgrid(us, vs)
    inside = (U >= 0) & (U < W) & (V >= 0) & (V < H)
    if not inside.any():
        return 1.0
    cov = np.zeros_like(U, bool)
    for nl, nt, nr, nb in near:
        c, w = (nl + nr) / 2, 0.3 * (nr - nl)
        cov |= (U >= c - w) & (U <= c + w) & (V >= nt) & (V <= nb)
    return float((cov & inside).sum() / inside.sum())


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def simulate(cams: dict, colour_cams: set, height_cams: set | None, seed: int, p: SimParams, out: Path,
             people: dict | None = None, model: dict | None = None) -> dict:
    """Simulate the court tracker's inputs for one layout and write them (CTIN2) to out/inputs.bin. Returns the
    truth: (frame, camera index, gid) -> 'team:number' (players) or 'ref', and per-camera checks."""
    from tools.court import FT, HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT
    import struct
    people = people or load_people()
    m = model or dict(np.load(OUT / "model.npz"))
    rng = np.random.default_rng(seed)
    pf_path = OUT / "factors.npz"
    pf = dict(np.load(pf_path)) if pf_path.exists() else {}
    a = float(m["facing_a"])
    g_mean = a + (1 - a) * 2 / np.pi
    pools = {t: m[f"hist_{t}"] for t in TEAM_NAMES}
    # each simulated player wears the colours of one real player of the team (their labelled crops): the same number
    # if the real roster has it, otherwise one of the team's players in turn; referees share the referee crops
    real = {t: sorted(k[8:] for k in m if k.startswith("hist_id_") and k[8:].startswith(t + ":")) for t in TEAM_NAMES}
    own = {}
    for q in range(len(people["team"])):
        t = people["team"][q]
        ident = f"{t}:{people['num'][q]}"
        own[q] = m[f"hist_id_{ident}"] if ident in real.get(t, []) else \
            m[f"hist_id_{real[t][q % len(real[t])]}"] if real.get(t) else pools[t]
    xmin, xmax, yhalf = -HALF_LEFT_FT * FT, HALF_RIGHT_FT * FT, WIDTH_FT * FT / 2
    names = sorted(cams)
    C = {c: _centre(cams[c]) for c in names}
    pid, fr_all, xy = people["pid"], people["frames"], people["xy"]
    team, num, height = people["team"], people["num"], people["height"]
    # per-person velocity (m/s, over +-5 frames) and facing
    vel = np.zeros_like(xy)
    for i in np.unique(pid):
        s = np.where(pid == i)[0]
        j0, j1 = np.clip(np.arange(len(s)) - 5, 0, len(s) - 1), np.clip(np.arange(len(s)) + 5, 0, len(s) - 1)
        dt = np.maximum(fr_all[s][j1] - fr_all[s][j0], 1) / 30
        vel[s] = (xy[s][j1] - xy[s][j0]) / dt[:, None]
    order = np.argsort(fr_all, kind="stable")
    bounds = np.searchsorted(fr_all[order], np.arange(fr_all.min(), fr_all.max() + 2))
    facing = rng.uniform(0, 2 * np.pi, pid.max() + 1)
    phase = rng.integers(0, REINFER, (pid.max() + 1, len(names)))
    noise = {}
    lean = np.zeros((pid.max() + 1, 2))
    # camera positions: cameras within 1.5 m of each other share one (a mount)
    mount = {}
    for cn in names:
        mount[cn] = next((mount[o] for o in mount if np.hypot(*(C[cn][:2] - C[o][:2])) < 1.5), len(set(mount.values())))
    los_err = np.zeros((pid.max() + 1, len(set(mount.values()))))
    why = collections.Counter()
    # calibration error fields: per camera and axis, three plane waves (wavelengths 10-30 m), RMS p.cal_err at 15 m
    field = {}
    for cn in names:
        r2 = np.random.default_rng([seed, sum(map(ord, cn))])
        wv = [(r2.uniform(0, 2 * np.pi), 2 * np.pi / r2.uniform(10, 30), r2.uniform(0, 2 * np.pi)) for _ in range(6)]
        field[cn] = wv
    def cal_offset(cn, x, y):
        if not p.cal_err:
            return np.zeros(2)
        d = np.hypot(x - C[cn][0], y - C[cn][1]) / 15
        w = field[cn]
        e = [sum(np.cos(k * (np.cos(a) * x + np.sin(a) * y) + ph) for a, k, ph in w[3 * ax:3 * ax + 3]) for ax in (0, 1)]
        return p.cal_err * d * np.array(e) / np.sqrt(1.5)
    recs = []            # (frame, cam index, gid, x, y, err_x, err_y, height, read, score, hist)
    truth = {}
    for fi, frame in enumerate(range(fr_all.min(), fr_all.max() + 1)):
        idx = order[bounds[fi]:bounds[fi + 1]]
        cams_out = []
        if len(idx):
            sp = np.hypot(*vel[idx].T)
            fast = sp > FAST
            facing[pid[idx[fast]]] = np.arctan2(vel[idx[fast], 1], vel[idx[fast], 0])
            facing[pid[idx[~fast]]] += rng.normal(0, 0.06, (~fast).sum())
            if p.los_sd:
                q_ = pid[idx]
                los_err[q_] = p.los_rho * los_err[q_] + np.sqrt(1 - p.los_rho ** 2) * p.los_sd * rng.normal(0, 1, (len(q_), los_err.shape[1]))
            if p.lean_sd:
                q_ = pid[idx]
                lean[q_] = p.lean_rho * lean[q_] + np.sqrt(1 - p.lean_rho ** 2) * p.lean_sd * rng.normal(0, 1, (len(q_), 2))
        for ci, cn in enumerate(names):
            P = cams[cn]
            dets = []
            if len(idx):
                box, dep = _boxes(P, xy[idx, 0], xy[idx, 1], height[pid[idx]], lean=lean[pid[idx]] if p.lean_sd else None)
                ok = dep > 0
                order_c = [j for j in np.argsort(dep) if ok[j]]
                cand = []
                for rank, j in enumerate(order_c):
                    l, t, r, b = box[j]
                    hpx = b - t
                    if hpx < p.min_px or not (0 <= (l + r) / 2 < W):
                        why[cn, "small or outside"] += 1
                        continue
                    in_img = (min(b, H) - max(t, 0)) / hpx
                    if in_img < p.min_in_image:
                        why[cn, "cut by the image border"] += 1
                        continue
                    near = [box[k] for k in order_c[:rank]]
                    vis = (1 - _covered(box[j], near, np.linspace(0.05, 0.95, 10))) * in_img
                    if vis < p.min_visible or rng.random() < p.miss:
                        why[cn, "hidden or missed"] += 1
                        continue
                    cand.append((vis, j, near))
                kept = []
                for vis, j, near in sorted(cand, key=lambda c: -c[0]):
                    if all(_iou(box[j], box[k]) <= p.nms_iou for k, _ in kept):
                        kept.append((j, near))
                    else:
                        why[cn, "NMS"] += 1
                for j, near in kept:
                    i = idx[j]
                    q = pid[i]
                    l, t, r, b = box[j]
                    hpx = b - t
                    e = noise.get((q, ci))
                    e = rng.normal(0, 1, 4) if e is None else p.noise_rho * e + np.sqrt(1 - p.noise_rho ** 2) * rng.normal(0, 1, 4)
                    noise[q, ci] = e
                    nl, nt, nr, nb = np.array([l, t, r, b]) + e * p.box_noise * hpx
                    cut = nt <= 5 or nb >= H - 5 or nl <= 2 or nr >= W - 2
                    u = (nl + nr) / 2
                    gx, gy = _plane_point(P, u, nb, 0.0)
                    on = xmin - COURT_MARGIN <= gx <= xmax + COURT_MARGIN and abs(gy) <= yhalf + COURT_MARGIN
                    if not on and nb < H - 5:
                        why[cn, "off court"] += 1
                        continue
                    feet_hidden = nb >= H - 5 or _covered(box[j], near, [0.92, 0.97]) > 0.5
                    fx, fy = _model_foot(P, xy[i], (nl, nt, nr, nb), feet_hidden) + cal_offset(cn, xy[i, 0], xy[i, 1])
                    if p.los_sd:
                        u_los = xy[i] - C[cn][:2]
                        fx, fy = np.array([fx, fy]) + los_err[q, mount[cn]] * u_los / 10.0
                    look = np.degrees(look_down(C[cn], xy[i, 0], xy[i, 1]))
                    gid = int(q) * 100 + ci
                    truth[frame, ci, gid] = "ref" if team[q] == "ref" else f"{team[q]}:{num[q]}"
                    read, score = None, 0.0
                    if num[q] and frame % REINFER == phase[q, ci] and hpx >= READ_MIN_PX and not cut and \
                            _covered(box[j], near, np.linspace(0.12, 0.62, 6)) < p.read_hidden:
                        los = np.arctan2(xy[i, 1] - C[cn][1], xy[i, 0] - C[cn][0])
                        g = a + (1 - a) * abs(np.cos(facing[q] - los))
                        ident = f"{team[q]}:{num[q]}"
                        gm = game_minute(frame)
                        fac = pf[ident][min(gm, len(pf[ident]) - 1)] if ident in pf else 1.0
                        rate = read_rate(m, hpx, tilt_deg(P, xy[i, 0], xy[i, 1])) * fac
                        if rng.random() < min(0.95, REINFER * p.read_scale * rate * g / g_mean):
                            read = str(num[q])
                            # wrong reads: the player's share in this game minute (if labelled), shrunk towards
                            # their whole-game share, itself shrunk towards the pooled share; values from this
                            # minute's misreads of the player first, then their other misreads
                            nr, nw = m.get(f"reads_{ident}", (0, 0))
                            p_w = (nw + K_WRONG * float(m["wrong"])) / (nr + K_WRONG)
                            nr2, nw2 = m.get(f"reads_{ident}@{gm}", (0, 0))
                            p_w = (nw2 + K_WRONG * p_w) / (nr2 + K_WRONG)
                            vals = m[f"wrongvals_{ident}@{gm}"] if nw2 and rng.random() < nw2 / (nw2 + 5) else \
                                m[f"wrongvals_{ident}"] if nw else []
                            if rng.random() < p_w:
                                kind = rng.choice(3, p=m["wrong_kinds"])
                                if len(vals) and rng.random() < nw / (nw + 5):
                                    kind = 3
                                if kind == 0 and len(read) == 2:
                                    read = read[rng.integers(2)]
                                elif kind == 1 and len(read) == 1:
                                    read = read + str(rng.integers(10))
                                elif kind == 3:      # what this player was actually misread as
                                    read = str(vals[rng.integers(len(vals))])
                                else:
                                    read = str(rng.choice([n for n in range(100) if str(n) != read]))
                                score = float(rng.choice(m["score_bad"]))
                            else:
                                score = float(rng.choice(m["score_ok"]))
                    hist, hgt = None, np.nan
                    if cn in colour_cams and frame % COLOUR_EVERY == 0 and not cut and nb - nt >= MIN_BOX_H \
                            and nb - nt >= 60 and nr - nl >= 15:
                        pool = own[q]
                        hist = pool[rng.integers(len(pool))].astype(np.float64)
                        o = _covered(box[j], near, np.linspace(0.2, 0.5, 4))
                        if o > p.colour_mix:
                            occ = next((k for k in reversed(range(len(near)))
                                        if _covered(box[j], [near[k]], np.linspace(0.2, 0.5, 4)) > 0), None)
                            if occ is not None:
                                pool_o = own[pid[idx[order_c[occ]]]]
                                hist = (1 - o) * hist + o * pool_o[rng.integers(len(pool_o))]
                        hist = (hist / hist.sum()).astype(np.float32)
                        if (cn in height_cams) if height_cams is not None else look <= HEIGHT_MAX_LOOK_DOWN:
                            hgt = _person_height(P, nl, nt, nr, nb)
                    recs.append((frame, ci, gid, fx, fy, fx - xy[i, 0], fy - xy[i, 1], hgt, read, score, hist, feet_hidden))
    # installation calibration: remove each camera's mean foot-point offset
    bias = {ci: np.mean([(r[5], r[6]) for r in recs if r[1] == ci], 0) if any(r[1] == ci for r in recs) else np.zeros(2)
            for ci in range(len(names))}
    out.mkdir(parents=True, exist_ok=True)
    sb = lambda s: struct.pack("<i", len(s.encode())) + s.encode()
    by_frame = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        by_frame[r[0]][r[1]].append(r)
    with open(out / "inputs.bin", "wb") as f:
        f.write(b"CTIN2" + struct.pack("<i", len(names)))
        for cn in names:
            f.write(struct.pack("<dd", C[cn][0], C[cn][1]) + sb(cn))
        for frame in range(fr_all.min(), fr_all.max() + 1):
            buf = [struct.pack("<ii", frame, len(names))]
            for ci in range(len(names)):
                rs = by_frame[frame][ci] if frame in by_frame else []
                buf.append(struct.pack("<ii", ci, len(rs)))
                for (_, _, gid, fx, fy, _, _, hgt, read, score, hist, _) in rs:
                    buf.append(struct.pack("<dddd", fx - bias[ci][0], fy - bias[ci][1], 0.0, hgt))
                    buf.append(struct.pack("<q", gid) + sb(""))
                    buf.append(struct.pack("<id", read is not None, score) + sb(read or ""))
                    buf.append(struct.pack("<i", hist is not None) + (hist.astype("<f4").tobytes() if hist is not None else b""))
            f.write(b"".join(buf))
        f.write(struct.pack("<i", -1))
    # checks against the measured pipeline
    chk = {"det_per_frame": {}, "reads": {}, "pair_d": collections.defaultdict(list)}
    nfr = fr_all.max() - fr_all.min() + 1
    for ci, cn in enumerate(names):
        rs = [r for r in recs if r[1] == ci]
        chk["det_per_frame"][cn] = len(rs) / nfr
        chk["reads"][cn] = sum(r[8] is not None for r in rs) / max(1, len(rs))
    per = collections.defaultdict(dict)
    for r in recs:
        if truth[r[0], r[1], r[2]] != "ref":
            per[r[0], r[2] // 100][r[1]] = (r[3] - bias[r[1]][0], r[4] - bias[r[1]][1])
    for (_, _), cs in per.items():
        ks = sorted(cs)
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                chk["pair_d"][names[ks[x]] + "-" + names[ks[y]]].append(np.hypot(*np.subtract(cs[ks[x]], cs[ks[y]])))
    chk["dropped"] = {f"{c} {k}": v / nfr for (c, k), v in sorted(why.items())}
    chk["err"] = {}
    for ci, cn in enumerate(names):
        e = np.array([np.hypot(r[5] - bias[ci][0], r[6] - bias[ci][1]) for r in recs if r[1] == ci])
        fh = np.array([r[11] for r in recs if r[1] == ci])
        chk["err"][cn] = [float(np.median(e)), float(np.percentile(e, 90)), float(fh.mean()),
                          float(np.median(e[fh])) if fh.any() else 0.0, float(np.median(e[~fh])) if (~fh).any() else 0.0]
    chk["bias"] = {names[ci]: b.round(3).tolist() for ci, b in bias.items()}
    return dict(truth=truth, checks=chk, names=names)


def moments(name: str) -> list:
    """The labelled moments of a GT minute, in the frames of the time-aligned clips the people come from."""
    from tools.gt_eval import frame_offsets
    t = json.loads((ROOT / "gt" / name / "tasks.json").read_text())
    off = frame_offsets(ROOT / "experiments" / "npct_r1").get("NL", 0)
    return [mm["frame"] - off for mm in t["moments"]]


def score_run(out: Path, truth: dict) -> dict:
    """Run the C++ court tracker on out/inputs.bin and score its identities against the truth."""
    import subprocess
    from tools.courttrack_export import load_cpp
    from tools.gt_eval import score
    from tools.world_tracker import identity_tracks
    subprocess.run([str(ROOT / "apps" / "courttrack" / "ct_replay"), str(out / "inputs.bin"), str(out / "cpp_out.txt")],
                   check=True, capture_output=True)
    det_ident, tracks = load_cpp(out / "cpp_out.txt")
    G = identity_tracks(tracks, det_ident)
    present = collections.defaultdict(set)    # frame -> people present (players)
    people = load_people()
    for f, q in zip(people["frames"], people["pid"]):
        if people["team"][q] != "ref":
            present[int(f)].add(int(q))
    last = int(people["frames"].max())
    sets = {"game_2to3": moments("game_2to3"), "game_4to5": moments("game_4to5"),
            "tuning": moments("game_2to3") + moments("game_4to5"), "held_out": moments("game_5to6"),
            "game": list(range(60, last + 1, 60))}
    res = {}
    for k, frames in sets.items():
        fs = set(frames)
        gt = {(f, str(c), str(g)): v for (f, c, g), v in truth.items() if f in fs and v != "ref"}
        s = score(G, gt)
        seen = {(f, g // 100) for (f, c, g), v in truth.items() if f in fs and v != "ref"}
        tot = sum(len(present[f]) for f in fs)
        res[k] = dict(idf1=s["idf1"], switches=s["switches"], det=len(seen) / max(1, tot))
    return res


def run_one(args) -> dict:
    letter, zw, seed, pkw, tag, keep = args
    cams, colour, heights = layout_cameras(letter, zw)
    out = OUT / "runs" / tag / f"{letter}{'' if zw is None else f'_{zw:.1f}'}_s{seed}"
    sim = simulate(cams, colour, heights, seed, SimParams(**pkw), out)
    res = score_run(out, sim["truth"])
    res["checks"] = sim["checks"]
    (out / "result.json").write_text(json.dumps(dict(params=pkw, result=res), indent=1, default=lambda o: dict(o)))
    if keep:   # inputs and truth for analysis
        with gzip.open(out / "truth.json.gz", "wt") as f:
            json.dump([[k[0], k[1], k[2], v] for k, v in sim["truth"].items()] + [["names", sim["names"]]], f)
    else:
        (out / "inputs.bin").unlink()
    return res


def summary(tags: list[str]) -> str:
    """Markdown table over the kept runs of the given tags: per layout and height, mean (range) over seeds."""
    import re
    groups = {}
    for tag in tags:
        for f in sorted((OUT / "runs" / tag).glob("*/result.json")):
            groups.setdefault(re.sub(r"_s\d+$", "", f.parent.name), []).append(json.loads(f.read_text())["result"])
    L = ["| Layout, height | Seeds | IDF1 2:00-3:00 | IDF1 4:00-5:00 | IDF1 held-out 5:00-6:00 | IDF1 whole game | "
         "Switches, whole game | Detections per frame | Digit reads per frame |", "|---|---|---|---|---|---|---|---|---|"]
    for k, rs in groups.items():
        cell = lambda s_: (f"{np.mean([r[s_]['idf1'] for r in rs]):.1%} ({min(r[s_]['idf1'] for r in rs):.0%}-"
                           f"{max(r[s_]['idf1'] for r in rs):.0%})")
        det = np.mean([sum(r["checks"]["det_per_frame"].values()) for r in rs])
        reads = np.mean([sum(r["checks"]["det_per_frame"][c] * r["checks"]["reads"][c] for c in r["checks"]["reads"])
                         for r in rs])
        L.append(f"| {k.replace('_', ' ')} m | {len(rs)} | {cell('game_2to3')} | {cell('game_4to5')} | {cell('held_out')} | "
                 f"{cell('game')} | {np.mean([r['game']['switches'] for r in rs]):.0f} | {det:.1f} | {reads:.2f} |")
    return "\n".join(L).replace("| A m |", "| A (current) |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("people", "stats", "sim", "summary"))
    ap.add_argument("layout", nargs="?", help="sim: layout letter (A, C, E, F)")
    ap.add_argument("--run", nargs="+", default=["up_r1"])
    ap.add_argument("--height", type=float, help="sim: mounting height of the new cameras (m)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--params", default="{}", help="sim: SimParams overrides as JSON")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--keep", action="store_true", help="sim: keep inputs.bin and the truth for analysis")
    ap.add_argument("--tags", nargs="+", default=["val4", "screen"], help="summary: run tags")
    ap.add_argument("--md", help="summary: write the table to this file")
    a = ap.parse_args()
    run = ROOT / "experiments" / a.run[0]
    if a.cmd == "sim":
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(a.layout, a.height, s, json.loads(a.params), a.tag, a.keep) for s in a.seeds]
        with ProcessPoolExecutor(min(len(jobs), 8)) as ex:
            for (_, _, seed, _, _, _), r in zip(jobs, ex.map(run_one, jobs)):
                print(f"seed {seed}: " + "; ".join(f"{k} IDF1 {v['idf1']:.1%} det {v['det']:.1%} sw {v['switches']}"
                                                   for k, v in r.items() if k != "checks"))
                c = r["checks"]
                print("  det/frame " + " ".join(f"{k} {v:.2f}" for k, v in c["det_per_frame"].items()) +
                      "; reads/det " + " ".join(f"{k} {v:.2%}" for k, v in c["reads"].items()))
                print("  pair distance median (90 %) " + " ".join(
                    f"{k} {np.median(v):.2f} ({np.percentile(v, 90):.2f})" for k, v in sorted(c["pair_d"].items())))
                print("  bias", c["bias"])
                print("  foot error median (90 %), feet-hidden share, median hidden / visible: " + " ".join(
                    f"{k} {v[0]:.2f} ({v[1]:.2f}) {v[2]:.0%} {v[3]:.2f}/{v[4]:.2f}" for k, v in c["err"].items()))
                print("  dropped per frame", {k: round(v, 2) for k, v in c["dropped"].items()})
    if a.cmd == "summary":
        txt = summary(a.tags)
        if a.md:
            Path(a.md).write_text(txt + "\n")
        print(txt)
    if a.cmd == "stats":
        st = measure([ROOT / "experiments" / r for r in a.run])
        m = fit(st)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "model.npz", **m)
        txt = summarise(st, m)
        ppl = load_people()
        src = ROOT / "experiments" / str(ppl["run"])
        pf = player_factors(src, ppl, m)
        np.savez_compressed(OUT / "factors.npz", **pf)
        txt += f"\nper-player read factors (observed / predicted reads per game minute, {src.name}):"
        for ident, v in sorted(pf.items()):
            txt += f"\n  {ident:10s} " + " ".join(f"{x:4.2f}" for x in v)
        txt += "\nper-player wrong reads at the labelled moments (reads, wrong, values):"
        for k in sorted(k for k in m if k.startswith("reads_") and "@" not in k):
            i = k[6:]
            txt += f"\n  {i:10s} {int(m[k][0]):4d} {int(m[k][1]):3d}  " + " ".join(sorted(map(str, m['wrongvals_' + i])))
        (OUT / "model.txt").write_text(f"runs: {' '.join(a.run)}\n" + txt + "\n")
        print(txt)
    if a.cmd == "people":
        P = build_people(run)
        for q in P:
            q["run"] = run.name
        save_people(P, OUT / "people.npz")
        for mi, c in people_check(P, run).items():
            print(f"{mi}: labelled detections on the right person {c['right'] / sum(c.values()):.1%}, wrong "
                  f"{c['wrong'] / sum(c.values()):.1%}, on none {c['no person'] / sum(c.values()):.1%}")
        n = collections.Counter()
        for p in P:
            n[p["team"]] += 1
        per = collections.Counter(int(f) for p in P for f in p["frames"])
        print(f"{len(P)} people ({dict(n)}); present per frame: median {np.median(list(per.values())):.0f}, "
              f"10-90 % {np.percentile(list(per.values()), 10):.0f}-{np.percentile(list(per.values()), 90):.0f}")
        for p in P:
            print(f"  {p['team']:7s} {p['num']:>3s}  frames {p['frames'][0]}-{p['frames'][-1]} ({len(p['frames'])})  "
                  f"height {p['height']:.2f}")


if __name__ == "__main__":
    main()
