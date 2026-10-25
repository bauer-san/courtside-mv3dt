"""Calibration check: refit cameras on the traced court lines, optionally with the camera centre held at a known mount.

Uses the courtcal prototype's line traces (inputs/courtcal_prototype.zip: *_feat.npy, distorted pixels; undistorted
here with the prototype's k1) and the measured court (tools/court.py, metres, world frame of the camera files).
Every traced feature is used: baseline, free-throw line, lane lines, free-throw circle, 3-point arc, half-court
line, sidelines. Residual = pixel distance from each traced point to the projected court line.

Fits per camera (pinhole, square pixels, principal point at the image centre, as the prototype):
  current      the camera file's matrix as it is
  free         focal length, rotation and centre refitted on all features
  mount        as free, with the centre held within --mount-box of MOUNT[cam] (the partner camera's centre)
Then cross-camera agreement on the labelled people (gt/*) with bbox-bottom floor points, current vs mount matrices.

  python -m tools.calib_check <run> [--mount-box 0.25] [--write-dir DIR]
"""

import argparse
import collections
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from tools import ROOT
from tools.court import FT, HALF_FLAG_FT, HALF_LOUNGE_FT, WIDTH_FT
from tools.stitch import floor_point, load_cameras

CX, CY = 960.0, 540.0
K1 = {"NL": -7.07e-08, "FL": -2.66e-08, "NR": -8.27e-08, "FR": -2.28e-08}   # courtcal prototype
LEFT = {"NL": +1, "FR": +1, "FL": -1, "NR": -1}                             # courtcal feature_map
FLAG_SIDE = {"NL": True, "FL": True, "NR": False, "FR": False}              # which key the camera's traces are of
MOUNT = {"FL": "NR", "FR": "NL"}                                            # far camera -> camera on the same mount


def undist(P: np.ndarray, k1: float) -> np.ndarray:
    d = np.asarray(P, float) - [CX, CY]
    u = d.copy()
    for _ in range(50):
        u = d / (1 + k1 * (u ** 2).sum(1, keepdims=True))
    return u + [CX, CY]


def traces(cam: str) -> dict:
    with zipfile.ZipFile(ROOT / "inputs" / "courtcal_prototype.zip") as z:
        with z.open(f"courtcal_proto/{cam}_feat.npy") as f:
            raw = np.load(f, allow_pickle=True).item()
    sgn, out = LEFT[cam], {}
    for k, P in raw.items():
        P = np.asarray(P, float)
        if P.ndim != 2 or len(P) < 6:
            continue
        if k == "LaneA":
            k = "lane+" if sgn > 0 else "lane-"
        elif k == "LaneB":
            k = "lane-" if sgn > 0 else "lane+"
        elif k in ("FTC", "FTCin"):
            k = "FTC"
        elif k in ("SideA", "SideB"):
            k = "side+" if (sgn > 0) == (k == "SideA") else "side-"
        out[k] = np.vstack([out[k], undist(P, K1[cam])]) if k in out else undist(P, K1[cam])
    return out


def court_features(flag: bool, step: float = 0.02) -> dict:
    """Dense floor samples (metres) of each feature of one key, plus half-court line and sidelines."""
    s = -1 if flag else 1
    h = (HALF_FLAG_FT if flag else HALF_LOUNGE_FT) * FT
    w2, lane, ftx = WIDTH_FT / 2 * FT, 6 * FT, s * (h - 19 * FT)
    b = s * h

    def seg(p, q, ext=0.0):
        p, q = np.array(p, float), np.array(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        p, q = p - ext * d, q + ext * d
        n = max(2, int(np.linalg.norm(q - p) / step))
        return np.linspace(p, q, n)

    th = np.linspace(0, 2 * np.pi, int(2 * np.pi * 6 * FT / step))
    bk = b - s * 5.25 * FT
    a = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi * 19.75 * FT / step))
    return {
        "Base": seg((b, -w2), (b, w2)),
        "FT": seg((ftx, -lane), (ftx, lane), 0.3),
        "lane+": seg((b, lane), (ftx, lane), 0.3),
        "lane-": seg((b, -lane), (ftx, -lane), 0.3),
        "FTC": np.c_[ftx + 6 * FT * np.cos(th), 6 * FT * np.sin(th)],
        "Arc": np.c_[bk - s * 19.75 * FT * np.cos(a), 19.75 * FT * np.sin(a)],
        "Half": seg((0, -w2), (0, w2), 0.5),
        "side+": seg((-HALF_FLAG_FT * FT, w2), (HALF_LOUNGE_FT * FT, w2)),
        "side-": seg((-HALF_FLAG_FT * FT, -w2), (HALF_LOUNGE_FT * FT, -w2)),
    }


def P_of(q: np.ndarray) -> np.ndarray:
    f = np.exp(q[0])
    R, _ = cv2.Rodrigues(q[1:4])
    C = q[4:7]
    K = np.array([[f, 0, CX], [0, f, CY], [0, 0, 1]])
    return K @ np.hstack([R, (-R @ C)[:, None]])


def q_of(P: np.ndarray) -> np.ndarray:
    """Decompose a 3x4 matrix into (log f, rvec, centre), forcing the prototype's intrinsics model."""
    M = P[:, :3] / np.cbrt(np.linalg.det(P[:, :3]))
    K, R = np.linalg.qr(np.linalg.inv(M))     # inv(M) = R^T K^-1
    K, R = np.linalg.inv(R), K.T
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    K = K / K[2, 2]
    if np.linalg.det(R) < 0:
        R = -R
    C = -np.linalg.solve(P[:, :3], P[:, 3])
    return np.r_[np.log((K[0, 0] + K[1, 1]) / 2), cv2.Rodrigues(R)[0].ravel(), C]


def residuals(P: np.ndarray, obs: dict, feats: dict) -> dict:
    out = {}
    for k, U in obs.items():
        X = feats[k]
        h = np.c_[X, np.zeros(len(X)), np.ones(len(X))] @ P.T
        ok = h[:, 2] > 0
        x = h[ok, :2] / h[ok, 2:3]
        if len(x) < 2:
            continue
        out[k] = cKDTree(x).query(U)[0]
    return out


def fit(q0: np.ndarray, obs: dict, feats: dict, centre=None, box=0.25) -> np.ndarray:
    """centre: hold the camera centre within +-box metres of it (hard bounds)."""
    def r(q):
        res = residuals(P_of(q), obs, feats)
        return np.concatenate([res[k] / np.sqrt(len(res[k])) * 10 for k in res])   # each feature weighted equally
    q = q0.copy()
    lo, hi = np.full(7, -np.inf), np.full(7, np.inf)
    if centre is not None:
        lo[4:7], hi[4:7] = centre - box, centre + box
        q[4:7] = np.clip(q[4:7], lo[4:7] + 1e-6, hi[4:7] - 1e-6)
        # re-aim the camera from the new centre before refining (keep the image centre on the same floor point)
        P0 = P_of(q0)
        g = np.linalg.solve(P0[:, [0, 1, 3]], [CX, CY, 1.0])
        tgt = np.r_[g[:2] / g[2], 0.0]
        z = tgt - q[4:7]
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        q[1:4] = cv2.Rodrigues(np.vstack([x, y, z]))[0].ravel()
    for _ in range(3):
        q = least_squares(r, q, loss="soft_l1", f_scale=2.0, x_scale="jac", bounds=(lo, hi)).x
    return q


def rms(res: dict) -> str:
    return " ".join(f"{k} {np.sqrt(np.mean(np.minimum(v, 30) ** 2)):.1f}" for k, v in sorted(res.items()))


def total(res: dict) -> float:
    return float(np.mean([np.sqrt(np.mean(np.minimum(v, 30) ** 2)) for v in res.values()]))


def agreement(run: Path, mats: dict) -> dict:
    """Median/share>0.9 m of cross-camera distance of labelled people, bbox-bottom floor points."""
    from tools.appearance import load_boxes
    from tools.xcam import labelled
    lab = {}
    for name in ("game_2to3", "game_4to5"):
        if (ROOT / "gt" / name / "labels").exists():
            lab.update(labelled(name))
    boxes = load_boxes(run, 60)
    box = {(fr, cam, str(g)): (l, t, r, b) for (cam, fr), v in boxes.items() for g, l, t, r, b in v}
    ppl = collections.defaultdict(list)
    for k, ident in lab.items():
        if k in box:
            l, t, r, b = box[k]
            if b < 1075:                       # feet inside the image
                ppl[(k[0], ident)].append((k[1], ((l + r) / 2, b)))
    pair = collections.defaultdict(list)
    for v in ppl.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                (ca, pa), (cb, pb) = v[i], v[j]
                if ca == cb:
                    continue
                xa, _ = floor_point(mats[ca], np.array([pa[0]]), np.array([pa[1]]))
                xb, _ = floor_point(mats[cb], np.array([pb[0]]), np.array([pb[1]]))
                pair["-".join(sorted((ca, cb)))].append(float(np.linalg.norm(xa[0] - xb[0])))
    allp = np.array([x for v in pair.values() for x in v])
    out = {"all": (float(np.median(allp)), float(np.mean(allp > 0.9)), len(allp))}
    for k, v in sorted(pair.items()):
        out[k] = (float(np.median(v)), float(np.mean(np.array(v) > 0.9)), len(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--mount-box", type=float, default=0.25, help="m: centre held within this of the mount")
    ap.add_argument("--write-dir", type=Path, help="write refitted 3x4 matrices (json) here; camera files untouched")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    cur = load_cameras(run / "camInfo_used")
    qcur = {c: q_of(P) for c, P in cur.items()}
    new = dict(cur)
    report = {}
    for cam in ("NL", "NR", "FL", "FR"):
        obs = traces(cam)
        feats = court_features(FLAG_SIDE[cam])
        obs = {k: v for k, v in obs.items() if k in feats}
        r0 = residuals(cur[cam], obs, feats)
        qf = fit(qcur[cam], obs, feats)
        rf = residuals(P_of(qf), obs, feats)
        line = {"current": (total(r0), rms(r0), qcur[cam]), "free": (total(rf), rms(rf), qf)}
        if cam in MOUNT:
            cm = qcur[MOUNT[cam]][4:7]
            qm = fit(qcur[cam], obs, feats, centre=cm, box=args.mount_box)
            rm = residuals(P_of(qm), obs, feats)
            line["mount"] = (total(rm), rms(rm), qm)
            new[cam] = P_of(qm)
        print(f"\n{cam}: traced features {', '.join(f'{k}({len(v)})' for k, v in sorted(obs.items()))}")
        for name, (tot, txt, q) in line.items():
            print(f"  {name:8s} mean RMS {tot:5.2f} px | centre {np.round(q[4:7], 2)} f {np.exp(q[0]):6.0f} | {txt}")
        report[cam] = {k: dict(rms_px=v[0], centre=v[2][4:7].tolist(), f=float(np.exp(v[2][0]))) for k, v in line.items()}
    print("\ncross-camera distance of labelled people (bbox-bottom floor points, feet in image): median m, share > 0.9 m, n")
    for name, mats in (("current", cur), ("FL/FR held to mount", new)):
        a = agreement(run, mats)
        print(f"  {name:20s} " + "  ".join(f"{k} {m:.2f}/{s:.0%}/{n}" for k, (m, s, n) in a.items()))
    if args.write_dir:
        args.write_dir.mkdir(parents=True, exist_ok=True)
        (args.write_dir / "matrices_mount.json").write_text(json.dumps({c: P.tolist() for c, P in new.items()}, indent=1))
        (args.write_dir / "calib_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
