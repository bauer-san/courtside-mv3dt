"""Re-trace a camera's court lines from the image itself and refit the camera (fixes bad hand traces).

1. clean court image: per-pixel median of --frames frames spread over the game (moving players disappear)
2. project the court model (tools/calib_check.court_features: standard HS/NCAA markings, measured court size) with
   the current matrix; along each projected line, search +-R px along the normal for the centre of a white paint
   stripe (ridge response: brighter than both sides, low saturation)
3. per feature keep only points consistent with one straight line (lines) or one ellipse (free-throw circle):
   lettering, logos and players next to a line are rejected
4. refit focal length, orientation (yaw/pitch/roll) and centre (tools/calib_check.fit) on the new traces; repeat
   with a narrower search
The 3-point arc is not used (its painted radius is uncertain). Writes overlays and, with --write, a camera file into
a separate dataset (never over the existing ones).

  python -m tools.court_refine <run> FL [--frames 25] [--write gym_full_flfix]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from tools import CAMERAS, ROOT
from tools.appearance import run_dataset
from tools.calib_check import FLAG_SIDE, P_of, court_features, fit, q_of, residuals, total, rms, traces
from tools.stitch import load_cameras

USE = ("Base", "FT", "lane+", "lane-", "FTC", "Half", "side+", "side-")


def clean_image(run: Path, cam: str, n: int, max_frame: int = 12600) -> np.ndarray:
    ds = run_dataset(run)
    cam_id = next(c for c, p, _ in CAMERAS if p == cam)
    cap = cv2.VideoCapture(str((ds / "videos" / f"{cam_id}_{cam}.mp4").resolve()))
    frames = []
    for f in np.linspace(300, max_frame, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def whiteness(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv[..., 2] * (1.0 - hsv[..., 1] / 255.0)        # bright and unsaturated


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    h = np.c_[X, np.zeros(len(X)), np.ones(len(X))] @ P.T
    ok = h[:, 2] > 0
    out = np.full((len(X), 2), np.nan)
    out[ok] = h[ok, :2] / h[ok, 2:3]
    return out


def search(W: np.ndarray, pts: np.ndarray, R: int, half: int = 3) -> np.ndarray:
    """For each polyline point, the ridge centre along the local normal within +-R px (NaN if none)."""
    H, Wd = W.shape
    out = np.full_like(pts, np.nan)
    d = np.gradient(pts, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    offs = np.arange(-R, R + 1)
    for i, (p, n) in enumerate(zip(pts, nrm)):
        if not np.all(np.isfinite(p)):
            continue
        s = p[None] + offs[:, None] * n[None]
        side = half + 3
        prof = []
        for k in (-side, 0, side):
            q = s + k * n
            x, y = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
            ok = (x >= 0) & (x < Wd) & (y >= 0) & (y < H)
            v = np.full(len(q), np.nan)
            v[ok] = W[y[ok], x[ok]]
            prof.append(v)
        resp = prof[1] - np.maximum(prof[0], prof[2])           # brighter than BOTH sides
        if not np.isfinite(resp).any():
            continue
        j = int(np.nanargmax(resp))
        if resp[j] > 25:
            out[i] = s[j]
    return out


def consistent(U: np.ndarray, circle: bool, tol: float = 2.0) -> np.ndarray:
    """RANSAC: points on one straight line (or one ellipse for the circle)."""
    U = U[np.all(np.isfinite(U), axis=1)]
    if len(U) < 10:
        return U[:0]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        if circle:
            idx = rng.choice(len(U), 6, replace=False)
            try:
                e = cv2.fitEllipse(U[idx].astype(np.float32))
            except cv2.error:
                continue
            (cx, cy), (a, b), ang = e
            if min(a, b) < 4:
                continue
            t = np.deg2rad(ang)
            x, y = U[:, 0] - cx, U[:, 1] - cy
            xr, yr = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
            r = np.sqrt((xr / (a / 2)) ** 2 + (yr / (b / 2)) ** 2)
            dist = np.abs(r - 1) * min(a, b) / 2
        else:
            i, j = rng.choice(len(U), 2, replace=False)
            dv = U[j] - U[i]
            if np.linalg.norm(dv) < 5:
                continue
            n = np.array([-dv[1], dv[0]]) / np.linalg.norm(dv)
            dist = np.abs((U - U[i]) @ n)
        inl = dist < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return U[best] if best is not None else U[:0]


def straightness(U: np.ndarray) -> float:
    m = U.mean(0)
    _, _, vt = np.linalg.svd(U - m)
    return float(np.sqrt(np.mean(((U - m) @ vt[1]) ** 2)))


def overlay(img: np.ndarray, P: np.ndarray, feats: dict, obs: dict, path: Path) -> None:
    out = img.copy()
    for k, X in feats.items():
        x = project(P, X)
        x = x[np.all(np.isfinite(x), axis=1)].astype(np.int32)
        if len(x) > 1:
            cv2.polylines(out, [x], False, (0, 255, 255) if k in USE else (255, 0, 255), 1, cv2.LINE_AA)
    for U in obs.values():
        for p in U[::3].astype(int):
            cv2.circle(out, tuple(p), 1, (0, 0, 255), -1)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("cam")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "calib_refine")
    ap.add_argument("--write", help="dataset name to write the refitted camera files into (copy of the run's)")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    cam = args.cam
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img = clean_image(run, cam, args.frames)
    cv2.imwrite(str(args.out_dir / f"{cam}_clean.jpg"), img)
    W = cv2.GaussianBlur(whiteness(img), (3, 3), 0)
    feats = court_features(FLAG_SIDE[cam], step=0.05)
    P = load_cameras(run / "camInfo_used")[cam]
    q = q_of(P)
    old = {k: v for k, v in traces(cam).items() if k in feats}
    r_old = residuals(P, old, feats)
    print(f"{cam} current calibration on the old traces: mean RMS {total(r_old):.2f} px | {rms(r_old)}")
    print("   old trace straightness (px): " + "  ".join(f"{k} {straightness(v):.2f}" for k, v in sorted(old.items())
                                                       if k not in ("FTC", "Arc")))
    for it, R in enumerate((25, 15, 8, 5)):
        Pc = P_of(q)
        obs = {}
        for k in USE:
            x = project(Pc, feats[k])
            hit = search(W, x, R)
            U = consistent(hit, circle=(k == "FTC"))
            if len(U) >= 20:
                obs[k] = U
        q = fit(q, obs, {k: feats[k] for k in obs})
        r = residuals(P_of(q), obs, feats)
        print(f"iteration {it + 1} (search +-{R} px): {sum(len(v) for v in obs.values())} points, mean RMS {total(r):.2f} px"
              f" | centre {np.round(q[4:7], 2)} f {np.exp(q[0]):.0f} | {rms(r)}")
    print("   new trace straightness (px): " + "  ".join(f"{k} {straightness(v):.2f}" for k, v in sorted(obs.items())
                                                       if k != "FTC"))
    overlay(img, P, feats, old, args.out_dir / f"{cam}_before.jpg")
    overlay(img, P_of(q), feats, obs, args.out_dir / f"{cam}_after.jpg")
    np.save(args.out_dir / f"{cam}_q.npy", q)
    np.save(args.out_dir / f"{cam}_traces.npy", obs, allow_pickle=True)      # image traces (undistorted px)
    print(args.out_dir / f"{cam}_after.jpg")
    if args.write:
        from tools.calib_joint import write_caminfo
        ds = ROOT / "datasets" / args.write
        ref = run_dataset(run)
        ds.mkdir(parents=True, exist_ok=True)
        for name in ("videos", "map.png", "transforms.yml", "manifest.json"):
            if not (ds / name).exists():
                os.symlink(os.path.relpath((ref / name).resolve(), ds), ds / name)
        qs = {c: q_of(Pc) for c, Pc in load_cameras(run / "camInfo_used").items()}
        qs[cam] = q
        write_caminfo(qs, run / "camInfo_used", ds / "camInfo")
        print(ds / "camInfo")


if __name__ == "__main__":
    main()
