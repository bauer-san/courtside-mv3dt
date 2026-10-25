"""Joint 4-camera calibration on the courtcal line traces with each far camera held within BOX of its mount partner
(FL with NR, FR with NL; user: pairs share one apparatus on top of each backboard, 6-10 in apart).

  python -m tools.calib_joint <run> <dataset-name>   # writes datasets/<name>/camInfo (standard HS/NCAA court)
"""
import numpy as np, cv2, json, sys
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tools import ROOT
import tools.calib_check as C
from tools.stitch import load_cameras
FT = 0.3048
run = ROOT / "experiments" / (sys.argv[1] if len(sys.argv) > 1 else "full_pn263_cf_jersey")
cur = load_cameras(run/"camInfo_used"); qcur = {c: C.q_of(P) for c, P in cur.items()}
OBS = {}
for cam in ("NL", "NR", "FL", "FR"):
    o = C.traces(cam); OBS[cam] = {k: v for k, v in o.items() if k in ("Base","FT","lane+","lane-","FTC","Arc","Half","side+","side-")}
# dims (metres): lane half width, FT line from baseline, FT circle r, arc r, basket from baseline, half_left, half_right, width/2
NAMES = ["lane_half", "ft_from_base", "ftc_r", "arc_r", "basket", "half_left", "half_right", "w2"]
def feats(left, d, step=0.03):
    lane, ftd, r, arc, bko, hf, hl, w2 = d
    s = -1 if left else 1; h = hf if left else hl; b = s * h; ftx = b - s * ftd; bk = b - s * bko
    def seg(p, q, ext=0.0):
        p, q = np.array(p, float), np.array(q, float); dd = (q - p) / np.linalg.norm(q - p); p, q = p - ext * dd, q + ext * dd
        return np.linspace(p, q, max(2, int(np.linalg.norm(q - p) / step)))
    th = np.linspace(0, 2 * np.pi, 200); a = np.linspace(-np.pi / 2, np.pi / 2, 400)
    # arc: circle about the basket clipped to the court width (corner lines ignored: not traced separately)
    ax, ay = bk - s * arc * np.cos(a), arc * np.sin(a); ok = np.abs(ay) <= w2
    return {"Base": seg((b, -w2), (b, w2)), "FT": seg((ftx, -lane), (ftx, lane), 0.3),
            "lane+": seg((b, lane), (ftx, lane), 0.3), "lane-": seg((b, -lane), (ftx, -lane), 0.3),
            "FTC": np.c_[ftx + r * np.cos(th), r * np.sin(th)], "Arc": np.c_[ax[ok], ay[ok]],
            "Half": seg((0, -w2), (0, w2), 0.5), "side+": seg((-hf, w2), (hl, w2)), "side-": seg((-hf, -w2), (hl, -w2))}
CAMS = ["NL", "NR", "FL", "FR"]
def unpack(x):
    d = x[:8]; q = {}
    for i, c in enumerate(["NL", "NR"]):
        q[c] = x[8 + 7 * i: 15 + 7 * i]
    for i, (c, m) in enumerate((("FL", "NR"), ("FR", "NL"))):
        y = x[22 + 7 * i: 29 + 7 * i]; q[c] = np.r_[y[:4], q[m][4:7] + y[4:7]]   # centre = partner + offset
    return d, q
def fixed_res(P, obs, F):
    out = {}
    for k, U in obs.items():
        X = F[k]; h = np.c_[X, np.zeros(len(X)), np.ones(len(X))] @ P.T; ok = h[:, 2] > 0
        if ok.sum() < 2:
            out[k] = np.full(len(U), 30.0); continue
        x = h[ok, :2] / h[ok, 2:3]; out[k] = np.minimum(cKDTree(x).query(U)[0], 60.0)
    return out
def resid(x, mount=True):
    d, q = unpack(x); out = []
    for c in CAMS:
        res = fixed_res(C.P_of(q[c]), OBS[c], feats(C.FLAG_SIDE[c], d))
        out += [res[k] / np.sqrt(len(res[k])) * 10 for k in sorted(res)]
    return np.concatenate(out)
def run_fit(d0, fix_dims=False, box=0.25):
    x0 = [*d0]
    for c in ("NL", "NR"): x0 += list(qcur[c])
    for c, m in (("FL", "NR"), ("FR", "NL")):
        qq = C.fit(qcur[c], OBS[c], C.court_features(C.FLAG_SIDE[c]), centre=qcur[m][4:7], box=box)
        x0 += list(qq[:4]) + list(qq[4:7] - qcur[m][4:7])
    x0 = np.array(x0, float); lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    if fix_dims: lo[:8], hi[:8] = x0[:8] - 1e-6, x0[:8] + 1e-6
    else:
        lo[:8] = [1.5, 5.0, 1.6, 5.5, 1.2, 9.0, 9.0, 6.0]; hi[:8] = [2.6, 6.2, 1.95, 7.4, 1.7, 12.5, 12.5, 8.0]
        x0[:8] = np.clip(x0[:8], lo[:8] + 1e-6, hi[:8] - 1e-6)
    for i in (26, 33): lo[i:i+3], hi[i:i+3] = -box, box     # mount offsets
    for i in (26, 33): x0[i:i+3] = np.clip(x0[i:i+3], -box + 1e-6, box - 1e-6)
    x = x0
    for _ in range(3): x = least_squares(resid, x, loss="soft_l1", f_scale=2.0, x_scale="jac", bounds=(lo, hi), max_nfev=400).x
    d, q = unpack(x)
    per = {c: C.total(C.residuals(C.P_of(q[c]), OBS[c], feats(C.FLAG_SIDE[c], d))) for c in CAMS}
    return d, q, per
cur_d = [6 * FT, 19 * FT, 6 * FT, 19.75 * FT, 5.25 * FT, 35.65 * FT, 35.51 * FT, 45.48 / 2 * FT]


def write_caminfo(q: dict, src_dir, out_dir) -> None:
    """Camera files like the originals (same person model), with the new matrices (normalised to P[2,3] = 1)."""
    import re
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(src_dir).glob("cam_*.yml")):
        cam = p.stem.split("_")[-1]
        P = C.P_of(q[cam])
        P = P / P[2, 3]
        txt = p.read_text()
        tail = txt[txt.index("# Person model"):]
        cen = q[cam][4:7]
        head = (f"# {cam} camera ({p.stem[:6]}) - joint calibration, far cameras held at their backboard mount\n"
                f"# (tools/calib_joint.py, standard HS/NCAA court markings; courtcal-proto-0.1 line traces)\n"
                "# World: metres, origin centre court, x toward right end, y to the left when facing the right end, z up.\n"
                "# Valid for UNDISTORTED 1920x1080 frames.\n"
                f"# Camera position (m): [{cen[0]:.3f}, {cen[1]:.3f}, {cen[2]:.3f}]  f {np.exp(q[cam][0]):.0f} px\n"
                "projectionMatrix_3x4_w2p:\n" + "".join(f"- {v:.10g}\n" for v in P.ravel()) + "\n")
        (out_dir / p.name).write_text(head + tail)


if __name__ == "__main__":
    import os
    from pathlib import Path
    ds = ROOT / "datasets" / sys.argv[2]
    d, q, per = run_fit(cur_d, fix_dims=True)
    print("line RMS px:", {c: round(v, 2) for c, v in per.items()})
    print("centres:", {c: np.round(q[c][4:7], 2).tolist() for c in CAMS})
    ref = ROOT / "datasets" / "gym_full"
    ds.mkdir(parents=True, exist_ok=True)
    for name in ("videos", "map.png", "transforms.yml", "manifest.json"):
        if not (ds / name).exists():
            os.symlink(os.path.relpath((ref / name).resolve(), ds), ds / name)
    write_caminfo(q, ref / "camInfo", ds / "camInfo")
    print(ds / "camInfo")
