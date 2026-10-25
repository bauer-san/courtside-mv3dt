"""Joint calibration of the four cameras on their physical mounts.

The cameras are two pairs on one apparatus each, on top of the backboards: FL with NR (right end), FR with NL (left
end), 6-10 in apart (user, 2026-10-23). Fit, on the image-based line traces of tools/court_refine.py
(experiments/calib_refine/<cam>_traces.npy, undistorted pixels):
  per camera  focal length, orientation (yaw, pitch, roll), centre; a far camera's centre = its partner's + an offset
              of at most --mount-box metres per axis
  court       left and right half-lengths and half-width are fitted (the old calibration's own measurement put the
              far cameras ~1 m behind their mounts); the markings are the standard HS/NCAA ones (lane 12 ft wide,
              free-throw line 19 ft from the baseline, circle radius 6 ft)
The same fit without the mount constraint is the reference for how much the constraint costs.

  python -m tools.calib_mount [--mount-box 0.25] [--write]   # --write: datasets/gym/camInfo + court size
"""

import argparse
import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from tools import ROOT
from tools.calib_check import P_of
from tools.court import FT, HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT

REFINE = ROOT / "experiments" / "calib_refine"
CAMS = ["NL", "NR", "FL", "FR"]
PARTNER = {"FL": "NR", "FR": "NL"}
FLAG_SIDE = {"NL": True, "FL": True, "NR": False, "FR": False}
LANE, FT_DIST, FTC_R = 6 * FT, 19 * FT, 6 * FT


def features(left: bool, hf: float, hl: float, w2: float, step: float = 0.04) -> dict:
    s = -1 if left else 1
    b = s * (hf if left else hl)
    ftx = b - s * FT_DIST

    def seg(p, q, ext=0.0):
        p, q = np.array(p, float), np.array(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        p, q = p - ext * d, q + ext * d
        return np.linspace(p, q, max(2, int(np.linalg.norm(q - p) / step)))

    th = np.linspace(0, 2 * np.pi, 240)
    return {"Base": seg((b, -w2), (b, w2)), "FT": seg((ftx, -LANE), (ftx, LANE), 0.3),
            "lane+": seg((b, LANE), (ftx, LANE), 0.3), "lane-": seg((b, -LANE), (ftx, -LANE), 0.3),
            "FTC": np.c_[ftx + FTC_R * np.cos(th), FTC_R * np.sin(th)],
            "Half": seg((0, -w2), (0, w2), 0.5), "side+": seg((-hf, w2), (hl, w2)), "side-": seg((-hf, -w2), (hl, -w2))}


def residuals_px(P: np.ndarray, obs: dict, F: dict) -> dict:
    out = {}
    for k, U in obs.items():
        X = F[k]
        h = np.c_[X, np.zeros(len(X)), np.ones(len(X))] @ P.T
        ok = h[:, 2] > 0
        if ok.sum() < 2:
            out[k] = np.full(len(U), 60.0)
            continue
        out[k] = np.minimum(cKDTree(h[ok, :2] / h[ok, 2:3]).query(U)[0], 60.0)
    return out


def unpack(x, mount: bool):
    hf, hl, w2 = x[:3]
    q = {}
    for i, c in enumerate(CAMS):
        v = x[3 + 7 * i: 10 + 7 * i]
        q[c] = v.copy()
    if mount:
        for c, m in PARTNER.items():
            q[c][4:7] = q[m][4:7] + q[c][4:7]          # stored as offset from the partner
    return (hf, hl, w2), q


def fit(obs: dict, q0: dict, dims0, mount: bool, box: float):
    x0 = list(dims0)
    for c in CAMS:
        v = q0[c].copy()
        if mount and c in PARTNER:
            v[4:7] = np.clip(v[4:7] - q0[PARTNER[c]][4:7], -box + 1e-6, box - 1e-6)
        x0 += list(v)
    x0 = np.array(x0, float)
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lo[:3], hi[:3] = [9.0, 9.0, 6.0], [12.5, 12.5, 8.0]
    if mount:
        for c in PARTNER:
            i = 3 + 7 * CAMS.index(c) + 4
            lo[i:i + 3], hi[i:i + 3] = -box, box

    def r(x):
        (hf, hl, w2), q = unpack(x, mount)
        e = []
        for c in CAMS:
            res = residuals_px(P_of(q[c]), obs[c], features(FLAG_SIDE[c], hf, hl, w2))
            e += [res[k] / np.sqrt(len(res[k])) * 10 for k in sorted(res)]
        return np.concatenate(e)

    x = x0
    for _ in range(3):
        x = least_squares(r, x, loss="soft_l1", f_scale=2.0, x_scale="jac", bounds=(lo, hi), max_nfev=300).x
    dims, q = unpack(x, mount)
    per = {}
    for c in CAMS:
        res = residuals_px(P_of(q[c]), obs[c], features(FLAG_SIDE[c], *dims))
        per[c] = (float(np.mean([np.sqrt(np.mean(np.minimum(v, 30) ** 2)) for v in res.values()])),
                  {k: float(np.sqrt(np.mean(np.minimum(v, 30) ** 2))) for k, v in res.items()})
    return dims, q, per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mount-box", type=float, default=0.25)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    obs = {c: np.load(REFINE / f"{c}_traces.npy", allow_pickle=True).item() for c in CAMS}
    q0 = {c: np.load(REFINE / f"{c}_q.npy") for c in CAMS}
    dims0 = (HALF_LEFT_FT * FT, HALF_RIGHT_FT * FT, WIDTH_FT * FT / 2)
    out = {}
    for name, mount in (("free", False), ("on mounts", True)):
        dims, q, per = fit(obs, q0, dims0, mount, args.mount_box)
        out[name] = (dims, q, per)
        print(f"\n{name}: court {2 * dims[2] / FT:.1f} ft wide, half-lengths left {dims[0] / FT:.1f} ft, "
              f"right {dims[1] / FT:.1f} ft (length {(dims[0] + dims[1]) / FT:.1f} ft)")
        for c in CAMS:
            C = q[c][4:7]
            print(f"  {c}: line RMS {per[c][0]:.2f} px  centre ({C[0]:+.2f}, {C[1]:+.2f}, {C[2]:.2f}) m  "
                  f"f {np.exp(q[c][0]):.0f} | " + " ".join(f"{k} {v:.1f}" for k, v in sorted(per[c][1].items())))
        for c, m in PARTNER.items():
            print(f"  {c}-{m} distance {np.linalg.norm(q[c][4:7] - q[m][4:7]):.2f} m")
    dims, q, per = out["on mounts"]
    (REFINE / "calib_mount.json").write_text(json.dumps(
        {"court_m": {"half_left": dims[0], "half_right": dims[1], "half_width": dims[2]},
         "cams": {c: {"q": q[c].tolist(), "line_rms_px": per[c][0]} for c in CAMS}}, indent=1))
    if args.write:
        write(dims, q, per)


def write(dims, q, per) -> None:
    """Make this the single calibration: camera files in datasets/gym/camInfo and the court size in
    tools/court.py (the previous files are archived next to them)."""
    import re
    import shutil
    dst = ROOT / "datasets" / "gym" / "camInfo"
    arch = ROOT / "datasets" / "gym" / "camInfo_archive_retraced_2026-10-23"
    if not arch.exists():
        shutil.copytree(dst, arch)
    for p in sorted(dst.glob("cam_*.yml")):
        cam = p.stem.split("_")[-1]
        P = P_of(q[cam])
        P = P / P[2, 3]
        txt = p.read_text()
        tail = txt[txt.index("# Person model"):]
        C = q[cam][4:7]
        head = (f"# {cam} camera ({p.stem[:6]}) - joint calibration on the physical mounts (tools/calib_mount.py)\n"
                "# Lines re-traced from the images (tools/court_refine.py); camera pairs held together on their\n"
                "# backboard apparatus (FL with NR, FR with NL, <= 0.25 m apart); standard HS/NCAA markings; court\n"
                "# length and width fitted (tools/court.py).\n"
                "# World: metres, origin centre court, x toward right end, y to the left when facing the right end, z up.\n"
                "# Valid for UNDISTORTED 1920x1080 frames.\n"
                f"# Adopted 2026-10-24 as the single calibration; court-line fit RMS {per[cam][0]:.2f} px.\n"
                f"# Camera position (m): [{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}]  f {np.exp(q[cam][0]):.0f} px\n"
                "projectionMatrix_3x4_w2p:\n" + "".join(f"- {v:.10g}\n" for v in P.ravel()) + "\n")
        p.write_text(head + tail)
    court = ROOT / "tools" / "court.py"
    s = court.read_text()
    s = re.sub(r"HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT = [^\n]+",
               f"HALF_LEFT_FT, HALF_RIGHT_FT, WIDTH_FT = {dims[0] / FT:.2f}, {dims[1] / FT:.2f}, {2 * dims[2] / FT:.2f}"
               "  # fitted with the cameras on their mounts (tools/calib_mount.py, 2026-10-24)", s)
    court.write_text(s)
    print("written", dst, "and", court)


if __name__ == "__main__":
    main()
