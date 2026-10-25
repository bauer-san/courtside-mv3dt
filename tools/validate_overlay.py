"""Phase 5: project court features through each camInfo matrix onto the matching video frame.

Draws on the undistorted 1920x1080 frame (the only resolution the matrices are valid for):
  yellow   court boundary (baselines + sidelines) and half-court line   [sidelines/half-court: measured, not fitted]
  green    lanes and FT circles                                           [fitted]
  magenta  3-point arcs
  cyan     check points: centre court, FT-line lane corners (A1/A2/B1/B2), court corners
  white    1.8 m person-model poles (SV3DT cylinder axis) at centre court and the FT-line lane corners

Also draws the same lines on map.png via transforms.yml to check the BEV transform.
Outputs experiments/overlays/<dataset>/{cam_XX_POS.jpg, grid.jpg, map.jpg}.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

from tools import CALIB_HEIGHT, CALIB_WIDTH, CAMERAS, DATASETS, ROOT
from tools.court import court_lines, court_points, project, world_to_map

COLOURS = {"boundary": (0, 255, 255), "half_court": (0, 255, 255), "lane": (0, 255, 0),
           "ft_circle": (0, 255, 0), "three_pt": (255, 0, 255)}
POLE_POINTS = ("centre", "A1", "A2", "B1", "B2")
POLE_HEIGHT = 1.8


def load_P(path: Path) -> np.ndarray:
    return np.array(yaml.safe_load(path.read_text())["projectionMatrix_3x4_w2p"], float).reshape(3, 4)


def read_frame(video: Path, frame: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"trim=start_frame={frame}:end_frame={frame + 1}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True,
    ).stdout
    img = np.frombuffer(raw, np.uint8)
    if img.size != CALIB_WIDTH * CALIB_HEIGHT * 3:
        raise ValueError(f"{video}: frame {frame} is not {CALIB_WIDTH}x{CALIB_HEIGHT}")
    return img.reshape(CALIB_HEIGHT, CALIB_WIDTH, 3).copy()


def densify(poly: np.ndarray, n: int = 40) -> np.ndarray:
    return np.vstack([np.linspace(a, b, n) for a, b in zip(poly[:-1], poly[1:])])


def draw_polyline(img, px, ok, colour, thickness=2):
    for i in range(len(px) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, tuple(np.int32(px[i])), tuple(np.int32(px[i + 1])), colour, thickness, cv2.LINE_AA)


def overlay_camera(img: np.ndarray, P: np.ndarray, label: str) -> np.ndarray:
    margin = 200
    for name, poly in court_lines():
        xy = densify(poly)
        px, depth = project(P, np.c_[xy, np.zeros(len(xy))])
        ok = (depth > 0) & (px[:, 0] > -margin) & (px[:, 0] < CALIB_WIDTH + margin) \
            & (px[:, 1] > -margin) & (px[:, 1] < CALIB_HEIGHT + margin)
        draw_polyline(img, px, ok, next(c for k, c in COLOURS.items() if name.startswith(k)))
    pts = court_points()
    for name, xy in pts.items():
        (u, v), = project(P, np.array([[*xy, 0.0]]))[0]
        if 0 <= u < CALIB_WIDTH and 0 <= v < CALIB_HEIGHT:
            cv2.circle(img, (int(u), int(v)), 7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, name, (int(u) + 9, int(v) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    for name in POLE_POINTS:
        x, y = pts[name]
        px, depth = project(P, np.array([[x, y, 0.0], [x, y, POLE_HEIGHT]]))
        if (depth > 0).all():
            cv2.line(img, tuple(np.int32(px[0])), tuple(np.int32(px[1])), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA)
    return img


def overlay_map(map_img: np.ndarray, T: np.ndarray) -> np.ndarray:
    for name, poly in court_lines():
        px = world_to_map(densify(poly), T)
        draw_polyline(map_img, px, np.ones(len(px), bool), (0, 160, 255), 1)
    for name, xy in court_points().items():
        (u, v), = world_to_map(xy[None], T)
        cv2.circle(map_img, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
    return map_img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASETS / "gym")
    ap.add_argument("--frame", type=int, default=3600)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "experiments" / "overlays" / args.dataset.name
    out.mkdir(parents=True, exist_ok=True)

    tiles = []
    for cam_id, pos, _ in CAMERAS:
        name = f"{cam_id}_{pos}"
        img = read_frame(args.dataset / "videos" / f"{name}.mp4", args.frame)
        img = overlay_camera(img, load_P(args.dataset / "camInfo" / f"{name}.yml"), f"{name}  frame {args.frame}")
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA))
        print(f"wrote {out / (name + '.jpg')}")
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out / "grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    T = np.array(yaml.safe_load((args.dataset / "transforms.yml").read_text())["T_ov2px"], float).reshape(3, 3)
    map_img = overlay_map(cv2.imread(str(args.dataset / "map.png")), T)
    cv2.imwrite(str(out / "map.jpg"), map_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {out / 'grid.jpg'}, {out / 'map.jpg'}")


if __name__ == "__main__":
    main()
