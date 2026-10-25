"""Clean-up steps: floor back-projection, duplicate merge, stitching."""

import numpy as np
import yaml

from tools import CAMERAS, DATASETS
from tools.court import project
from tools.stitch import Params, Track, floor_point, merge_duplicates, stitch


def track(f0, f1, xy0, xy1, gid):
    fr = np.arange(f0, f1 + 1)
    return Track(fr, np.linspace(xy0, xy1, len(fr)), {"NL"}, [gid])


def test_floor_point_inverts_projection():
    for cam_id, pos, _ in CAMERAS:
        P = np.array(yaml.safe_load((DATASETS / "gym" / "camInfo" / f"{cam_id}_{pos}.yml").read_text())
                     ["projectionMatrix_3x4_w2p"]).reshape(3, 4)
        xy = np.array([[0.0, 0.0], [3.0, -2.0], [-5.0, 4.0]])
        px, depth = project(P, np.c_[xy, np.zeros(3)])
        back, ok = floor_point(P, px[:, 0], px[:, 1])
        np.testing.assert_allclose(back[depth > 0], xy[depth > 0], atol=1e-6)
        assert (ok == (depth > 0)).all()


def test_merge_duplicates_joins_close_overlapping_tracks():
    t = {"a": track(0, 99, (0, 0), (5, 0), "a"),
         "b": track(20, 80, (1.0, 0.3), (4.0, 0.3), "b"),     # same person, other camera, 0.3 m off
         "c": track(0, 99, (0, 3), (5, 3), "c")}               # different person, 3 m away
    out = merge_duplicates(t, Params())
    assert len(out) == 2
    assert sorted(len(x.members) for x in out.values()) == [1, 2]


def test_stitch_links_plausible_gap_only():
    t = {"a": track(0, 49, (0, 0), (2, 0), "a"),
         "b": track(60, 99, (2.5, 0), (4, 0), "b"),            # 10-frame gap, 0.5 m: link
         "c": track(55, 99, (9, 5), (9, 5), "c")}              # 7+ m away after 6 frames: no link
    out = stitch(t, Params())
    assert len(out) == 2
    joined = next(x for x in out.values() if len(x.members) == 2)
    assert joined.members == ["a", "b"] and joined.start == 0 and joined.end == 99
