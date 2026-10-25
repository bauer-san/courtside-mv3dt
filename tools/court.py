"""Court geometry in world metres (origin centre court, +x right end, +y to the left when facing the right end, z up).

Dimensions follow courtcal_prototype (floor_v2.py / overlay_court.py): measured half-lengths
(left end 35.65 ft, right end 35.51 ft), width 45.48 ft, 12 ft lane, FT line 19 ft from the
baseline, FT circle r = 6 ft, basket centre 5.25 ft from the baseline, 3-pt arc r = 19.75 ft
with straight corner segments. Only lanes, FT lines, FT circles and baselines were fitted;
sidelines and the half-court line are measured checks.
"""

import numpy as np

FT = 0.3048
HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT = 35.16, 35.12, 45.05  # fitted with the cameras on their mounts (tools/calib_mount.py, 2026-10-24)

# World (m) -> layout_v2.png pixel (1 px = 1 inch).
T_WORLD_TO_MAP = np.array([[39.3701, 0.0, 499.8], [0.0, -39.3701, 344.9], [0.0, 0.0, 1.0]])


def _ft(pts) -> np.ndarray:
    return np.asarray(pts, dtype=float) * FT


def court_lines(n_arc: int = 120) -> list[tuple[str, np.ndarray]]:
    """Named polylines on the floor (z = 0), each an (N, 2) array in metres."""
    hf, hl, w2 = HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT / 2
    lines = [
        ("boundary", _ft([(-hf, -w2), (hl, -w2), (hl, w2), (-hf, w2), (-hf, -w2)])),
        ("half_court", _ft([(0, -w2), (0, w2)])),
    ]
    th = np.linspace(0, 2 * np.pi, n_arc)
    for s, h in ((-1, hf), (1, hl)):  # s = -1 left end (A), +1 right end (B)
        b = s * h
        ft_x = b - s * 19
        bk = b - s * 5.25
        lines.append((f"lane_{'A' if s < 0 else 'B'}", _ft([(b, -6), (ft_x, -6), (ft_x, 6), (b, 6)])))
        lines.append((f"ft_circle_{'A' if s < 0 else 'B'}", _ft(np.c_[ft_x + 6 * np.cos(th), 6 * np.sin(th)])))
        a = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
        arc = np.c_[bk - s * 19.75 * np.cos(a), 19.75 * np.sin(a)]
        lines.append((f"three_pt_{'A' if s < 0 else 'B'}", _ft(np.vstack([[b, -19.75], arc, [b, 19.75]]))))
    return lines


def court_points() -> dict[str, np.ndarray]:
    """Named check points (metres): centre court, FT-line lane corners, baseline/sideline corners."""
    hf, hl, w2 = HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT / 2
    pts = {"centre": (0, 0)}
    for e, s, h in (("A", -1, hf), ("B", 1, hl)):
        b = s * h
        pts[f"{e}1"] = (b - s * 19, 6)    # FT-line lane corner, +y side
        pts[f"{e}2"] = (b - s * 19, -6)   # FT-line lane corner, right side
        pts[f"{e}_base+"] = (b, w2)
        pts[f"{e}_base-"] = (b, -w2)
    return {k: _ft(v) for k, v in pts.items()}


def project(P: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points with a 3x4 matrix. Returns (N, 2) pixels and (N,) depth sign."""
    h = np.c_[xyz, np.ones(len(xyz))] @ P.T
    return h[:, :2] / h[:, 2:3], h[:, 2]


def world_to_map(xy: np.ndarray, T: np.ndarray = T_WORLD_TO_MAP) -> np.ndarray:
    h = np.c_[xy, np.ones(len(xy))] @ T.T
    return h[:, :2] / h[:, 2:3]
