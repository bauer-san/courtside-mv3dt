"""Phase 2 checks: clip command construction and generated clip datasets (skipped if absent)."""

import json

import pytest

from tools import CAMERAS, DATASETS
from tools.make_clips import encode_cmd

CLIP_DATASETS = {"courtside_60s": 1800, "courtside_180s": 5400}


def test_encode_cmd_selects_by_frame_number(tmp_path):
    cmd = encode_cmd(tmp_path / "in.mp4", tmp_path / "out.mp4", start=3600, count=1800)
    vf = cmd[cmd.index("-vf") + 1]
    assert "trim=start_frame=3600:end_frame=5400" in vf
    assert "setpts=PTS-STARTPTS" in vf
    assert "-ss" not in cmd  # time-based seeking would break frame sync
    assert "-an" in cmd


@pytest.mark.parametrize("name,frames", CLIP_DATASETS.items())
def test_clip_dataset(name, frames):
    ds = DATASETS / name
    if not (ds / "manifest.json").exists():
        pytest.skip(f"{name} not generated (run scripts/phase2_clips.sh)")
    manifest = json.loads((ds / "manifest.json").read_text())
    assert manifest["frames"] == frames
    for cam_id, pos, _ in CAMERAS:
        cam = manifest["cameras"][cam_id]
        assert cam["frames"] == frames
        assert cam["sync"]["ok"]
        assert (ds / "videos" / f"{cam_id}_{pos}.mp4").is_file()
        # Calibration is shared with the base dataset, not copied.
        caminfo = ds / "camInfo"
        assert caminfo.is_symlink()
        assert (caminfo / f"{cam_id}_{pos}.yml").read_text() == \
            (DATASETS / "courtside" / "camInfo" / f"{cam_id}_{pos}.yml").read_text()
