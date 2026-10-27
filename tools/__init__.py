"""Gym MV3DT workspace tools."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
DATASETS = ROOT / "datasets"

# The game: 0:00-6:29 of the videos (frames up to 6:29 at 30 fps); a warm-up follows, which is out of scope.
GAME_END = 6 * 1800 + 29 * 30

# Fixed stream order. Index = DeepStream source id.
CAMERAS = [
    ("cam_00", "NL", "00_NL"),
    ("cam_01", "FL", "01_FL"),
    ("cam_02", "NR", "02_NR"),
    ("cam_03", "FR", "03_FR"),
]

# Projection matrices are valid only at this resolution (undistorted pixels).
CALIB_WIDTH, CALIB_HEIGHT = 1920, 1080
