"""Config patcher keeps layout and edits only the targeted keys."""

from tools.ds_config import set_ini_key, set_tracker_key

TRACKER = """%YAML:1.0
TargetManagement:
  maxTargetsPerStream: 200
  outputTerminatedTracks: 0
PoseEstimator:
  poseEstimatorType: 1
  # comment
  poseInferenceInterval: 30
"""

INI = """[sink2]
enable=1
type=3 # file
output-file=outVideos/tiled_display_raw.mp4

[streammux]
width=1920
"""


def test_tracker_replace_and_insert():
    t = set_tracker_key(TRACKER, "TargetManagement", "outputTerminatedTracks", "1")
    t = set_tracker_key(t, "TargetManagement", "terminatedTrackFilename", "/x/track_dump")
    t = set_tracker_key(t, "PoseEstimator", "poseEstimatorType", "0")
    assert "  outputTerminatedTracks: 1" in t
    assert "  terminatedTrackFilename: /x/track_dump" in t
    assert t.index("terminatedTrackFilename") < t.index("PoseEstimator:")
    assert "  poseEstimatorType: 0" in t and "# comment" in t and "poseInferenceInterval: 30" in t


def test_tracker_new_section():
    t = set_tracker_key(TRACKER, "NewSection", "k", "v")
    assert t.endswith("NewSection:\n  k: v\n")


def test_ini_replace():
    s = set_ini_key(INI, "sink2", "output-file", "outVideos/osd.mp4")
    assert "output-file=outVideos/osd.mp4" in s and "tiled_display_raw" not in s
    assert "type=3 # file" in s and "[streammux]\nwidth=1920" in s
