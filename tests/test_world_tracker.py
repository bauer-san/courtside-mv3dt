"""Court-plane tracker on synthetic detections."""

import numpy as np

from tools.world_tracker import WorldTracker, WTParams, estimate_bias


def run(frames):
    wt = WorldTracker(WTParams())
    for f, dets in enumerate(frames):
        wt.step(f, {c: np.array(v) for c, v in dets.items()})
    return wt.result()


def walk(f, x0, v):
    return (x0 + v * f / 30.0, 0.0)


def test_two_cameras_one_person_gives_one_track():
    rng = np.random.default_rng(0)
    frames = []
    for f in range(150):
        x, y = walk(f, -3.0, 2.0)
        frames.append({"NL": [(x + rng.normal(0, 0.1), y + rng.normal(0, 0.1))],
                       "FL": [(x + 0.4 + rng.normal(0, 0.1), y + rng.normal(0, 0.1))]})  # 0.4 m camera offset
    tracks = run(frames)
    assert len(tracks) == 1
    assert len(tracks[0].out) == 150


def test_handover_through_gap_keeps_id():
    frames = []
    for f in range(200):
        x, y = walk(f, -5.0, 3.0)
        if f < 80:
            frames.append({"NL": [(x, y)]})
        elif f < 110:
            frames.append({})            # 1 s in nobody's view
        else:
            frames.append({"FR": [(x, y)]})
    tracks = run(frames)
    assert len(tracks) == 1
    assert tracks[0].out[0][0] == 0 and tracks[0].out[-1][0] == 199


def test_crossing_players_keep_separate_tracks():
    frames = []
    for f in range(120):
        a = (-3.0 + 4.0 * f / 30.0 * 0.5, 1.0)
        b = (3.0 - 4.0 * f / 30.0 * 0.5, -1.0)   # pass each other 2 m apart
        frames.append({"FL": [a, b]})
    tracks = run(frames)
    assert len(tracks) == 2
    ends = sorted(t.out[-1][2] for t in tracks)
    assert ends[0] < 0 < ends[1]                 # each kept its own side (y = -1 / +1)


def test_estimate_bias_recovers_offset():
    by_frame = {}
    for f in range(100):
        by_frame[f] = {"NL": [(f * 0.01, 0.0)], "FL": [(f * 0.01 + 0.4, 0.2)]}
    b = estimate_bias(by_frame)
    np.testing.assert_allclose(np.subtract(b["FL"], b["NL"]), (0.4, 0.2), atol=1e-6)
    np.testing.assert_allclose(np.add(b["FL"], b["NL"]), (0.0, 0.0), atol=1e-6)


def test_gate_matrix_matches_scalar_gate():
    # the scalar gate is the isotropic, plain-Mahalanobis definition
    rng = np.random.default_rng(1)
    wt = WorldTracker(WTParams(aniso=False, cost_logdet=False))
    for f in range(20):
        wt.step(f, {"FL": rng.uniform(-3, 3, (5, 2))})
    Z = rng.uniform(-3, 3, (6, 2))
    C = wt._gate_matrix(Z, 21)
    ref = np.array([[wt._gate(t, z, 21) for z in Z] for t in wt.tracks])
    np.testing.assert_allclose(np.where(np.isfinite(ref), ref, -1), np.where(np.isfinite(C), C, -1), rtol=1e-9)


def test_aniso_noise_is_long_along_the_line_of_sight():
    wt = WorldTracker(WTParams(), cam_xy={"FR": np.array([-12.0, 0.0]), "NR": np.array([0.0, -10.0])})
    for f in range(20):
        wt.step(f, {"FR": np.array([[0.0, 0.0]])})
    # 1 m off along FR's line of sight (x) is cheaper than 1 m across it (y)
    C = wt._gate_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]), 20, "FR")
    assert C[0, 0] < C[0, 1]
    # two cameras at right angles: each one's good (tangential) direction wins in the fusion
    z, R = wt._fuse([("FR", np.array([0.8, 0.1])), ("NR", np.array([0.1, -0.8]))])
    assert abs(z[0]) < 0.3 and abs(z[1]) < 0.3
    assert np.all(np.linalg.eigvalsh(R) < WTParams().sigma_tangential ** 2 * 1.01)


def test_number_swap_is_split_and_identity_relinked():
    # one track follows #7, the person leaves view, the track then wrongly continues on #9 (swap);
    # later #7 reappears: the swap must be split and #7's new track re-linked to its old identity
    reads = {}
    frames = []
    for f in range(300):
        dets, keys = [], []
        if f < 100:
            pos, num = (-3.0 + f * 0.02, 0.0), "7"
        elif f < 200:
            pos, num = (-1.0 + (f - 100) * 0.02, 0.2), "9"     # contiguous: the tracker keeps the same track
        else:
            pos, num = None, None
        if pos:
            dets.append(pos)
            keys.append((f, "FL", "g"))
            if f % 5 == 0:
                reads[(f, "FL", "g")] = (num, 0.95)
        if f >= 210:                                             # #7 seen again elsewhere
            dets.append((4.0, 2.0))
            keys.append((f, "FR", "h"))
            if f % 5 == 0:
                reads[(f, "FR", "h")] = ("7", 0.95)
        frames.append((dets, keys))
    wt = WorldTracker(WTParams(), reads)
    for f, (dets, keys) in enumerate(frames):
        by_cam = {}
        for d, k in zip(dets, keys):
            by_cam.setdefault(k[1], ([], []))
            by_cam[k[1]][0].append(d)
            by_cam[k[1]][1].append(("red", np.nan, 0.95, k))      # same team: identity = team + number
        wt.step(f, {c: np.array(v[0]) for c, v in by_cam.items()}, {c: v[1] for c, v in by_cam.items()})
    tracks = wt.result()
    by_num = {t.number(WTParams()): t for t in tracks}
    assert set(by_num) == {"7", "9"}
    seven = by_num["7"]
    assert seven.out[0][0] == 0 and seven.out[-1][0] == 299       # #7 re-linked across the swap and the gap
    assert by_num["9"].out[0][0] >= 100                            # the #9 part was split off


def test_encounter_exchange_swaps_histories_before_the_encounter():
    wt = WorldTracker(WTParams())
    for f in range(40):
        wt.step(f, {"FL": np.array([[-2.0 + 0.01 * f, 0.0], [2.0 - 0.01 * f, 0.0]])})
    a, b = sorted(wt.result(), key=lambda t: t.out[0][1])
    ta, tb = a.tid, b.tid
    wt._exchange_before(a, b, 20)
    assert a.tid == tb and b.tid == ta
    assert a.out[0][1] > 0 and a.out[-1][1] < 0          # b's early positions, then a's own
    assert all(o[0] < 20 for o in a.out if o[1] > 0) and len({o[0] for o in a.out}) == len(a.out)


def _identity_run(p, reads_from, frames=300, second=None):
    """One red player walking along x; reads of reads_from(f) every 5 frames. second: optional (x, y) of another
    confirmed player for a frame range (a close encounter)."""
    reads = {(f, "FL", "a"): (reads_from(f), 0.95) for f in range(0, frames, 5) if reads_from(f)}
    wt = WorldTracker(p, reads)
    for f in range(frames):
        dets, meta = [(-3.0 + 0.01 * f, 0.0)], [("red", np.nan, 0.95, (f, "FL", "a"))]
        if second:
            dets.append(second(f))
            meta.append(("red", np.nan, 0.95, (f, "FL", "b")))
        wt.step(f, {"FL": np.array(dets)}, {"FL": meta})
    return wt


def test_identity_from_recent_reads_follows_a_number_change():
    # after frame 150 the track follows #9, read with an occasional misread (every third read "1"): never three
    # identical reads in a row, so the tracker's swap split does not fire; the recent reads still favour 9
    reads = lambda f: "7" if f < 150 else ("1" if (f // 5) % 3 == 0 else "9")
    old = _identity_run(WTParams(id_recent=False, split_majority=False, player_appearance=False), reads, frames=600)
    new = _identity_run(WTParams(id_recent=True, split_majority=False, player_appearance=False), reads, frames=600)
    assert old.det_ident[(599, "FL", "a")] == "red:7"          # all reads since the start: 7 still counts
    assert new.det_ident[(599, "FL", "a")] == "red:9"          # the last 10 s: 9


def test_identity_doubted_after_close_encounter_until_confirmed():
    reads = lambda f: "7" if f < 100 or f >= 250 else None
    # a second player walks past within 0.5 m around frames 150-200, then leaves
    other = lambda f: (-3.0 + 0.01 * f, 0.5 if 150 <= f < 200 else 4.0)
    wt = _identity_run(WTParams(id_doubt=True, player_appearance=False), reads, second=other)
    assert wt.det_ident[(120, "FL", "a")] == "red:7"
    assert wt.det_ident[(220, "FL", "a")] is None              # after the encounter, no read yet
    assert wt.det_ident[(260, "FL", "a")] == "red:7"           # confirmed again


def test_majority_split_survives_a_misread():
    # #7 for 150 frames, then the track follows #9 read with every third read a misread: the default split (three
    # identical reads in a row) never fires; the majority split does, at about frame 150
    reads = lambda f: "7" if f < 150 else ("1" if (f // 5) % 3 == 0 else "9")
    run = lambda p: _identity_run(p, reads, frames=300)
    base = run(WTParams(split_majority=False, player_appearance=False))
    maj = run(WTParams(split_majority=True, player_appearance=False))
    first = lambda wt: sorted(t.out[0][0] for t in wt.result())
    assert first(base) == [0]
    assert len(first(maj)) == 2 and 150 <= first(maj)[1] <= 240
