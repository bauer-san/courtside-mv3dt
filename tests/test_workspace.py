"""Phase 1 checks: dataset layout and camInfo consistency with court_calibration.json."""

import json

import numpy as np
import pytest
import yaml

from tools import CALIB_HEIGHT, CALIB_WIDTH, CAMERAS, DATASETS, INPUTS

DATASET = DATASETS / "courtside"
ARCHIVE = DATASET / "camInfo_archive_courtcal_proto_2026-10-21"   # the original calibration (court_calibration.json)


def load_caminfo_P(path):
    data = yaml.safe_load(path.read_text())
    vals = data["projectionMatrix_3x4_w2p"]
    assert len(vals) == 12
    return np.array(vals, dtype=float).reshape(3, 4)


def project(P, xyz):
    p = P @ np.append(xyz, 1.0)
    return p[:2] / p[2]


@pytest.fixture(scope="module")
def calibration():
    return {s["id"]: s for s in json.loads((INPUTS / "court_calibration.json").read_text())["sensors"]}


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_video_links_resolve(cam_id, pos, _):
    link = DATASET / "videos" / f"{cam_id}_{pos}.mp4"
    assert link.is_symlink(), "videos must be linked, not copied"
    assert link.resolve().is_file()
    assert link.resolve().name.endswith("_undist.mp4")


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_archived_caminfo_matches_calibration(cam_id, pos, _, calibration):
    P = load_caminfo_P(ARCHIVE / f"{cam_id}_{pos}.yml")
    sensor = calibration[cam_id]
    assert sensor["position"] == pos
    assert (sensor["attributes"]["frameWidth"], sensor["attributes"]["frameHeight"]) == (CALIB_WIDTH, CALIB_HEIGHT)
    ref = np.array(sensor["cameraMatrix"], dtype=float)
    ref /= ref[2, 3]
    np.testing.assert_allclose(P, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_archived_camera_centre_matches_calibration(cam_id, pos, _, calibration):
    P = load_caminfo_P(ARCHIVE / f"{cam_id}_{pos}.yml")
    # Camera centre is the right null vector of P.
    c = np.linalg.svd(P)[2][-1]
    np.testing.assert_allclose(c[:3] / c[3], calibration[cam_id]["cameraPosition_m"], atol=1e-3)


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_centre_court_in_front_of_camera(cam_id, pos, _):
    P = load_caminfo_P(DATASET / "camInfo" / f"{cam_id}_{pos}.yml")
    assert (P @ np.array([0.0, 0.0, 0.0, 1.0]))[2] > 0


@pytest.mark.parametrize("cam_id,pos,_", CAMERAS)
def test_adopted_caminfo_provenance_and_sanity(cam_id, pos, _, calibration):
    # the adopted calibration (tools/court_refine.py) says so, and stays near the original camera position
    path = DATASET / "camInfo" / f"{cam_id}_{pos}.yml"
    assert "Adopted 2026-10-24" in path.read_text() and "physical mounts" in path.read_text()
    centre = lambda cid, p: (lambda c: c[:3] / c[3])(np.linalg.svd(load_caminfo_P(
        DATASET / "camInfo" / f"{cid}_{p}.yml"))[2][-1])
    partner = {"FL": ("cam_02", "NR"), "FR": ("cam_00", "NL")}
    if pos in partner:      # far cameras share a backboard apparatus with a near camera (6-10 in apart)
        assert np.linalg.norm(centre(cam_id, pos) - centre(*partner[pos])) < 0.45
    else:
        assert np.linalg.norm(centre(cam_id, pos) - np.array(calibration[cam_id]["cameraPosition_m"])) < 0.5
