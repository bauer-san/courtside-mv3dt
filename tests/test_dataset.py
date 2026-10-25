"""Phase 5 checks: court model, BEV transform, and the assembled gym dataset."""

import csv

import numpy as np
import pytest
import yaml

from tools import CAMERAS, DATASETS, INPUTS
from tools.build_dataset import PERSON_CLASS_IDS
from tools.court import T_WORLD_TO_MAP, court_points, project, world_to_map

DATASET = DATASETS / "gym"


@pytest.fixture(scope="module")
def alignment_points():
    with open(INPUTS / "alignment_points_v2.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def test_court_points_geometry():
    # alignment_points_v2.csv used the original court size; since 2026-10-24 the length/width are fitted
    # (tools/calib_mount.py), so check the markings against the court's own baselines instead
    from tools.court import FT, HALF_LEFT_FT, HALF_RIGHT_FT
    pts = court_points()
    np.testing.assert_allclose(pts["A1"], [-(HALF_LEFT_FT - 19) * FT, 6 * FT], atol=1e-6)
    np.testing.assert_allclose(pts["A2"], [-(HALF_LEFT_FT - 19) * FT, -6 * FT], atol=1e-6)
    np.testing.assert_allclose(pts["B1"], [(HALF_RIGHT_FT - 19) * FT, 6 * FT], atol=1e-6)
    np.testing.assert_allclose(pts["B2"], [(HALF_RIGHT_FT - 19) * FT, -6 * FT], atol=1e-6)


def test_map_transform_matches_alignment_csv(alignment_points):
    for r in alignment_points.values():
        uv = world_to_map(np.array([[float(r["x_m"]), float(r["y_m"])]]))[0]
        np.testing.assert_allclose(uv, [float(r["map_px_x"]), float(r["map_px_y"])], atol=1.0)


def test_transforms_yml_and_map():
    t = yaml.safe_load((DATASET / "transforms.yml").read_text())
    np.testing.assert_allclose(np.array(t["T_ov2px"]).reshape(3, 3), T_WORLD_TO_MAP)
    assert (DATASET / "map.png").resolve() == (INPUTS / "layout_v2.png").resolve()


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_caminfo_model_info(cam_id, pos, _):
    info = yaml.safe_load((DATASET / "camInfo" / f"{cam_id}_{pos}.yml").read_text())
    assert sorted(m["classID"] for m in info["modelInfo"]) == list(PERSON_CLASS_IDS)
    assert len(info["projectionMatrix_3x4_w2p"]) == 12


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_some_court_points_visible(cam_id, pos, _):
    P = np.array(yaml.safe_load((DATASET / "camInfo" / f"{cam_id}_{pos}.yml").read_text())
                 ["projectionMatrix_3x4_w2p"]).reshape(3, 4)
    xy = np.array(list(court_points().values()))
    px, depth = project(P, np.c_[xy, np.zeros(len(xy))])
    inside = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < 1920) & (px[:, 1] >= 0) & (px[:, 1] < 1080)
    assert inside.sum() >= 3
