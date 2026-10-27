"""Camera placement study: coverage, image resolution, viewing angle and multi-view geometry of candidate layouts.

Each camera: position (m, court frame of datasets/gym/camInfo), aim point on the floor, horizontal field of view
(1920 x 1080, no roll). On a 0.5 m grid over the court (+0.5 m margin), for a standing player 1.8 m tall:
  sees        feet and head both inside the image, in front of the camera
  px          player height in pixels (the reader needs >= READ_PX, the detector >= DETECT_PX)
  elevation   angle the camera looks down at the player's torso (steeper -> fewer occlusions); "shadow" = length on
              the floor hidden behind a 1.8 m player = 1.8 / tan(elevation of the head ray)
  diversity   largest angle between the floor directions of two cameras that see the point readable
Summary per layout: share of court area with >= 2 readable views, with >= 2 readable views >= 45 deg apart, with
>= 3 readable views; median best elevation and shadow; compute per frame set from the measured stream cost.

  python -m tools.camera_layout [--validate base_r1] [--out reports/figures/camera_layouts.png] [--md out.md]
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from tools import GAME_END, ROOT
from tools.court import court_lines
from tools.evaluate import X_MAX, X_MIN, Y_HALF
from tools.stitch import load_cameras

W, H = 1920, 1080
PERSON = 1.8
READ_PX, DETECT_PX = 150, 80
GRID = 0.5


def look_at(pos, target, hfov_deg) -> np.ndarray:
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return K @ np.hstack([R, (-R @ pos)[:, None]])


def centre(P) -> np.ndarray:
    return -np.linalg.solve(P[:, :3], P[:, 3])


def project(P, X):
    h = np.c_[X, np.ones(len(X))] @ P.T
    return h[:, :2] / h[:, 2:3], h[:, 2]


def grid():
    xs = np.arange(X_MIN - 0.5 + GRID / 2, X_MAX + 0.5, GRID)
    ys = np.arange(-Y_HALF - 0.5 + GRID / 2, Y_HALF + 0.5, GRID)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    inside = (gx >= X_MIN) & (gx <= X_MAX) & (np.abs(gy) <= Y_HALF)
    return gx, gy, inside


def per_camera(P, gx, gy) -> dict:
    pts = np.c_[gx.ravel(), gy.ravel()]
    feet, df = project(P, np.c_[pts, np.zeros(len(pts))])
    head, dh = project(P, np.c_[pts, np.full(len(pts), PERSON)])
    ok = (df > 0) & (dh > 0)
    for p in (feet, head):
        ok &= (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    px = np.linalg.norm(feet - head, axis=1)
    C = centre(P)
    d = np.hypot(pts[:, 0] - C[0], pts[:, 1] - C[1])
    elev = np.degrees(np.arctan2(C[2] - PERSON / 2, d))
    head_el = np.arctan2(np.maximum(C[2] - PERSON, 1e-3), d)
    shadow = np.where(C[2] > PERSON, PERSON / np.tan(head_el), np.inf)
    direction = np.arctan2(pts[:, 1] - C[1], pts[:, 0] - C[0])
    shape = gx.shape
    return {k: v.reshape(shape) for k, v in dict(sees=ok, px=np.where(ok, px, 0), elev=elev, shadow=shadow,
                                                 dir=direction).items()}


def evaluate(cams: dict) -> dict:
    gx, gy, inside = grid()
    pc = {c: per_camera(P, gx, gy) for c, P in cams.items()}
    readable = np.stack([pc[c]["sees"] & (pc[c]["px"] >= READ_PX) for c in cams])
    detect = np.stack([pc[c]["sees"] & (pc[c]["px"] >= DETECT_PX) for c in cams])
    n_read, n_det = readable.sum(0), detect.sum(0)
    dirs = np.stack([pc[c]["dir"] for c in cams])
    div = np.zeros(gx.shape)
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a = np.abs(np.angle(np.exp(1j * (dirs[i] - dirs[j]))))
            a = np.degrees(np.minimum(a, np.pi - a + np.pi * 0))       # 0..180
            both = readable[i] & readable[j]
            div = np.where(both, np.maximum(div, a), div)
    elev = np.stack([np.where(readable[i], pc[c]["elev"], -1) for i, c in enumerate(cams)]).max(0)
    shadow = np.stack([np.where(readable[i], pc[c]["shadow"], np.inf) for i, c in enumerate(cams)]).min(0)
    m = inside
    return dict(
        n_read=n_read, n_det=n_det, div=div, elev=elev, inside=inside, gx=gx, gy=gy,
        detect_any=float((n_det[m] >= 1).mean()), read2=float((n_read[m] >= 2).mean()),
        read2_div=float(((n_read >= 2) & (div >= 45))[m].mean()), read3=float((n_read[m] >= 3).mean()),
        read0=float((n_read[m] == 0).mean()),
        elev_med=float(np.median(elev[m & (elev > 0)])) if (m & (elev > 0)).any() else 0.0,
        shadow_med=float(np.median(shadow[m & np.isfinite(shadow)])) if (m & np.isfinite(shadow)).any() else np.inf,
    )


# --- candidate layouts -----------------------------------------------------------------------------------------
# Venue assumptions (to be replaced by measurements): walls 2.0 m beyond the lines, usable mounting height up to
# 6.0 m on walls; backboard mounts as now (x = +-9.65 m, z = 4.1 m).
HF, HL, W2 = -X_MIN, X_MAX, Y_HALF
WALL = 2.0
ZW = 6.0


def current_layout():
    return {c: P for c, P in load_cameras(ROOT / "datasets" / "gym" / "camInfo").items()}


def layouts() -> dict:
    """Candidate layouts; new cameras get their lens and aim from optimise(). Returns name -> (cams, choices)."""
    cur = current_layout()
    side = {"S+": (0, W2 + WALL, ZW), "S-": (0, -W2 - WALL, ZW)}
    corners = {f"C{i}": (sx * (HL + WALL), sy * (W2 + WALL), ZW)
               for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))}
    quarter = {f"Q{i}": (sx * 5.4, sy * (W2 + WALL), ZW) for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1)))}
    base = {"B+": (HL + WALL, 0, ZW), "B-": (-HF - WALL, 0, ZW)}
    far = {k: cur[k] for k in ("FL", "FR")}
    L = {"A. current: 4 on the backboards (as calibrated)": (cur, {})}
    L["B. current 4 + 2 high on the sideline walls at half court"] = optimise(side, cur)
    L["C. 4 high in the corners"] = optimise(corners, {})
    L["D. 4 high in the corners + 2 high sideline walls at half court"] = optimise({**corners, **side}, {})
    L["E. 2 far backboard cameras + 2 high sideline walls at half court"] = optimise(side, far)
    L["F. 4 high on the sidelines at the quarter lines + 2 high on the baselines"] = optimise({**quarter, **base}, {})
    return L


def optimise(spec: dict, fixed: dict, passes: int = 3) -> dict:
    """spec: name -> position of each new camera. Coordinate descent over lens field of view and aim point,
    maximising area with >= 2 readable views >= 45 deg apart (+ 0.25 x area with >= 1 readable view);
    `fixed` cameras (projection matrices) stay as they are."""
    hfovs = (50, 60, 70, 80, 95)
    targets = [(x, y, 0.0) for x in (-8, -4, 0, 4, 8) for y in (-3, 0, 3)]
    cand = {n: [(h, t, look_at(pos, t, h)) for h in hfovs for t in targets
                if np.hypot(t[0] - pos[0], t[1] - pos[1]) > 1.0] for n, pos in spec.items()}
    choice = {n: cand[n][len(cand[n]) // 2] for n in spec}

    def score(cams):
        r = evaluate(cams)
        m = r["inside"]
        return r["read2_div"] + 0.25 * float((r["n_read"][m] >= 1).mean())

    best = score({**fixed, **{n: c[2] for n, c in choice.items()}})
    for _ in range(passes):
        for n in spec:
            for c in cand[n]:
                trial = {**fixed, **{k: (c if k == n else v)[2] for k, v in choice.items()}}
                s = score(trial)
                if s > best + 1e-9:
                    best, choice[n] = s, c
    return {**fixed, **{n: c[2] for n, c in choice.items()}}, {n: (c[0], c[1][:2]) for n, c in choice.items()}


def validate(run: Path) -> list:
    """Predicted player height in pixels vs detected boxes (MV3DT foot positions, current calibration)."""
    from tools.appearance import load_boxes
    from tools.stitch import load_detections
    cams = load_cameras(run / "camInfo_used")
    d = load_detections(run / "kafka.jsonl.gz")
    boxes = load_boxes(run, 30)
    box = {(fr, cam, str(g)): b for (cam, fr), v in boxes.items() for g, *b in v}
    out = []
    for cam, P in cams.items():
        ratios = []
        sel = np.where((d["cam"] == cam) & (d["frame"] % 30 == 0) & (d["frame"] <= GAME_END))[0]
        for i in sel:
            b = box.get((int(d["frame"][i]), cam, str(d["gid"][i])))
            if b is None or b[3] >= H - 2 or b[1] <= 2:
                continue
            x, y = d["x"][i], d["y"][i]
            if not (X_MIN <= x <= X_MAX and abs(y) <= Y_HALF):
                continue
            (f0, h0), _ = project(P, np.array([[x, y, 0.0], [x, y, PERSON]]))
            pred = np.linalg.norm(np.array(f0) - np.array(h0))
            ratios.append((b[3] - b[1]) / pred)
        out.append((cam, len(ratios), float(np.median(ratios)), float(np.percentile(ratios, 25)),
                    float(np.percentile(ratios, 75))))
    return out


def figure(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    names = list(results)
    cmap = ListedColormap(["#d9d7d2", "#b9d3f0", "#6aa3e6", "#2a78d6", "#174f94"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    cols = 2
    rows = (len(names) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.9 * rows), facecolor="#fcfcfb")
    for ax, n in zip(axes.ravel(), names):
        r = results[n]
        ax.set_facecolor("#fcfcfb")
        im = ax.pcolormesh(r["gx"][:, 0], r["gy"][0], np.minimum(r["n_read"], 4).T, cmap=cmap, norm=norm, shading="nearest")
        bad = (r["n_read"] >= 2) & (r["div"] < 45)
        ax.contourf(r["gx"][:, 0], r["gy"][0], bad.T.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["///"])
        for _, pts in court_lines(40):
            ax.plot(pts[:, 0], pts[:, 1], color="#3b3a37", lw=0.6)
        for c, C in r["centres"].items():
            ax.plot(C[0], C[1], marker="o", ms=5, color="#eb6834", mec="#0b0b0b", mew=0.5, clip_on=False)
        ax.set_aspect("equal")
        ax.set_xlim(X_MIN - 2.6, X_MAX + 2.6)
        ax.set_ylim(-Y_HALF - 2.6, Y_HALF + 2.6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        import textwrap
        ax.set_title("\n".join(textwrap.wrap(n, 52)) + f"\n{r['ncam']} streams | >=2 readable views {r['read2']:.0%}"
                     f" | >=2 views 45 deg apart {r['read2_div']:.0%}", fontsize=9, loc="left", color="#0b0b0b")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.45, wspace=0.08, bottom=0.1)
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3, 4])
    cb.ax.set_xticklabels(["0", "1", "2", "3", "4+"])
    cax.set_title(f"cameras seeing a standing player at >= {READ_PX} px (jersey-readable)", fontsize=9)
    fig.text(0.01, 0.0, "Hatched: >= 2 readable views but from directions < 45 deg apart (poor for separating "
             "overlapping players). Orange dots: camera positions.", fontsize=8.5, color="#52514e")
    fig.savefig(path, dpi=120, facecolor="#fcfcfb", bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", default="base_r1")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "figures" / "camera_layouts.png")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    if args.validate:
        print("validation: detected box height / predicted standing height (current cameras)")
        for cam, n, med, q1, q3 in validate(ROOT / "experiments" / args.validate):
            print(f"  {cam}: n={n}  median {med:.2f}  (IQR {q1:.2f}-{q3:.2f})")
    res = {}
    for name, (cams, choices) in layouts().items():
        print(name, {k: f"{h} deg -> aim ({t[0]:.0f}, {t[1]:.0f})" for k, (h, t) in choices.items()}, flush=True)
        r = evaluate(cams)
        r["ncam"] = len(cams)
        r["centres"] = {c: centre(P) for c, P in cams.items()}
        res[name] = r
    lines = ["| layout | streams | detectable anywhere | >=2 readable views | >=2 readable, >=45 deg apart | "
             ">=3 readable views | no readable view | median best look-down angle | median shadow behind a player |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        lines.append(f"| {n} | {r['ncam']} | {r['detect_any']:.0%} | {r['read2']:.0%} | {r['read2_div']:.0%} | "
                     f"{r['read3']:.0%} | {r['read0']:.0%} | {r['elev_med']:.0f} deg | {r['shadow_med']:.1f} m |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.md:
        args.md.write_text(md)
    figure(res, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
