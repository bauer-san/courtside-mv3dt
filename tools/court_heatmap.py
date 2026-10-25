"""Where on the court identities break: heat maps of court-tracker events, per player-minute of occupancy.

Over the game (frames <= --max-frame) with the court tracker (online relink, team colour, no numbers by default):
  occupancy     player-minutes per cell (confirmed tracks' positions on frames with detections)
  new tracks    confirmations that were not re-linked to a lost track (a new identity)
  relinks       confirmations re-linked to a lost track (a fragment repaired)
  lost          confirmed tracks that ended by coasting out
  duplicates    frames where two confirmed tracks with detections from disjoint cameras are within 1.5 m
  coverage      cameras that see a person standing in the cell (feet and a 1.8 m head inside the image)
plus the ground-truth switches (tools/gt_causes.py) and cross-camera disagreement of the labelled pairs
(tools/xcam.py, MV3DT foot points). Rates are per player-minute; cells with < --min-occ player-minutes are blank.

  python -m tools.court_heatmap <run> [--max-frame 12600] [--cell 2.0] [--out gt/game_2to3/court_heatmap.png]
"""

import argparse
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import ROOT
from tools.court import court_lines
from tools.evaluate import X_MAX, X_MIN, Y_HALF
from tools.gt_causes import analyse
from tools.gt_eval import load_gt
from tools.stitch import load_cameras
from tools.world_tracker import WTParams, track_run
from tools.xcam import camera_centre

FPS = 30
CAUSE_STYLE = {"swap": ("X", "#d62728"), "duplicate": ("o", "#1f77b4"), "drifted": ("^", "#ff7f0e"),
               "takeover-swap": ("P", "#9467bd"), "takeover-dup": ("s", "#2ca02c"), "lost/reborn": ("D", "#7f7f7f")}


def grid(cell: float):
    xe = np.arange(X_MIN, X_MAX + cell, cell)
    ye = np.arange(-Y_HALF, Y_HALF + cell, cell)
    return xe, ye


def hist(xy: np.ndarray, xe, ye, w=None) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((len(xe) - 1, len(ye) - 1))
    return np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye], weights=w)[0]


def draw_court(ax, cams):
    for _, pts in court_lines(60):
        ax.plot(pts[:, 0], pts[:, 1], color="0.35", lw=0.7, zorder=3)
    done = set()
    for c, C in cams.items():                 # cameras sharing a mount get one marker and a joint label
        if c in done:
            continue
        mates = [k for k, D in cams.items() if np.linalg.norm(D[:2] - C[:2]) < 0.6]
        done.update(mates)
        ax.plot(*C[:2], marker="v" if C[0] < 0 else "^", color="k", ms=5, zorder=4, clip_on=False)
        ax.annotate("+".join(sorted(mates)), C[:2], xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=7, zorder=4, annotation_clip=False)
    ax.set_xlim(X_MIN - 1.2, X_MAX + 1.2)
    ax.set_ylim(-Y_HALF - 1.2, Y_HALF + 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def coverage(cams: dict, xe, ye, zs=(0.0, 1.8)) -> np.ndarray:
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    out = np.zeros((len(xc), len(yc)))
    for P in cams.values():
        for i, x in enumerate(xc):
            for j, y in enumerate(yc):
                ok = True
                for z in zs:
                    h = P @ np.array([x, y, z, 1.0])
                    u, v = h[:2] / h[2]
                    ok &= h[2] > 0 and 0 <= u < 1920 and 0 <= v < 1080
                out[i, j] += ok
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--max-frame", type=int, default=12600)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--min-occ", type=float, default=0.5)
    ap.add_argument("--no-numbers", action="store_true", help="court tracker without jersey numbers")
    ap.add_argument("--gt", default="game_2to3")
    ap.add_argument("--out", type=Path, default=ROOT / "gt" / "game_2to3" / "court_heatmap.png")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    cams = load_cameras(run / "camInfo_used")
    centres = {c: camera_centre(P) for c, P in cams.items()}
    p = WTParams(online_relink=True, use_numbers=not args.no_numbers)
    tracks = track_run(run, p)
    events = [e for e in track_run.last_events if e[1] <= args.max_frame]
    xe, ye = grid(args.cell)

    occ_xy = np.array([o[1:3] for t in tracks for o in t.out if o[0] <= args.max_frame])
    occ = hist(occ_xy, xe, ye) / FPS / 60                     # player-minutes
    valid = occ >= args.min_occ

    def rate(kind):
        xy = np.array([e[2:4] for e in events if e[0] == kind]).reshape(-1, 2)
        return np.where(valid, hist(xy, xe, ye) / np.maximum(occ, 1e-9), np.nan), len(xy)

    # duplicates: per frame, pairs of confirmed tracks with detections, disjoint cameras, within 1.5 m
    per_frame = collections.defaultdict(list)
    for t in tracks:
        for f, x, y, _ in t.out:
            if f <= args.max_frame:
                per_frame[f].append((x, y, t.cams.get(f, frozenset())))
    dup_xy = []
    for f, v in per_frame.items():
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if not (v[a][2] & v[b][2]) and np.hypot(v[a][0] - v[b][0], v[a][1] - v[b][1]) < 1.5:
                    dup_xy.append(((v[a][0] + v[b][0]) / 2, (v[a][1] + v[b][1]) / 2))
    dup = np.where(valid, hist(np.array(dup_xy).reshape(-1, 2), xe, ye) / FPS / 60 / np.maximum(occ, 1e-9), np.nan)

    panels = [("occupancy (player-minutes per cell)", occ, "Greys", None)]
    for kind, title in (("new", "new identities"), ("lost", "tracks lost (coasted out)"),
                        ("relink", "relinks (fragments repaired)")):
        r, n = rate(kind)
        panels.append((f"{title}: {n} in game, per player-minute", r, "magma_r", None))
    panels.append(("duplicate tracks: share of player time", dup, "magma_r", None))
    whole = coverage(cams, xe, ye)
    panels.append(("cameras seeing the feet (number: feet / whole person)", coverage(cams, xe, ye, (0.0,)), "viridis", (0, 4)))

    fig, axes = plt.subplots(4, 2, figsize=(13, 17))
    axes = axes.ravel()
    for ax, (title, M, cmap, lim) in zip(axes, panels):
        vmin, vmax = lim if lim else (0, np.nanpercentile(M, 97) if np.isfinite(M).any() else 1)
        im = ax.pcolormesh(xe, ye, M.T, cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        draw_court(ax, centres)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        if "cameras" in title:
            for i in range(len(xe) - 1):
                for j in range(len(ye) - 1):
                    ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, f"{int(M[i, j])}/{int(whole[i, j])}", ha="center",
                            va="center", fontsize=7, color="w", zorder=5)

    # ground-truth switches and cross-camera disagreement (2:00-3:00)
    gt = load_gt(args.gt, run)
    tg = track_run(run, p)
    tid = {t.tid: t for t in tg}
    r = analyse(tg, gt)
    ax = axes[6]
    ax.pcolormesh(xe, ye, occ.T, cmap="Greys", vmin=0, vmax=np.nanpercentile(occ, 97) * 1.6, zorder=1)
    draw_court(ax, centres)
    cnt = collections.Counter()
    for e in r["events"]:
        B = tid[e["B"]]
        o = min(B.out, key=lambda o: abs(o[0] - e["f2"]))
        mk, col = CAUSE_STYLE[e["cause"]]
        ax.scatter(o[1], o[2], marker=mk, c=col, s=40, edgecolors="k", linewidths=0.4, zorder=6,
                   label=e["cause"] if e["cause"] not in cnt else None)
        cnt[e["cause"]] += 1
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    ax.set_title(f"ground-truth switches, 2:00-3:00 ({sum(cnt.values())}), at the player's position", fontsize=10)

    ax = axes[7]
    from tools.xcam import labelled
    from tools.stitch import load_detections
    lab = labelled(args.gt, run)
    d = load_detections(run / "kafka.jsonl.gz")
    idx = {(int(f), str(c), str(g)): i for i, (f, c, g) in enumerate(zip(d["frame"], d["cam"], d["gid"]))}
    ppl = collections.defaultdict(list)
    for k, ident in lab.items():
        if k in idx:
            i = idx[k]
            ppl[(k[0], ident)].append((k[1], np.array([d["x"][i], d["y"][i]])))
    pts, dist = [], []
    for v in ppl.values():
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if v[a][0] != v[b][0]:
                    pts.append((v[a][1] + v[b][1]) / 2)
                    dist.append(np.linalg.norm(v[a][1] - v[b][1]))
    draw_court(ax, centres)
    pts = np.array(pts)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=dist, cmap="RdYlGn_r", vmin=0, vmax=1.8, s=28, edgecolors="k",
                    linewidths=0.3, zorder=6)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="m")
    ax.set_title(f"cross-camera disagreement of labelled pairs (n={len(dist)}; > 0.9 m makes duplicates)", fontsize=10)

    var = "default: team colour + jersey numbers" if not args.no_numbers else "team colour, no numbers"
    fig.suptitle(f"{run.name}, court tracker ({var}), game frames 0-{args.max_frame}\n"
                 "flag end left (NL+FR mount), lounge end right (NR+FL mount)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(args.out)
    # numeric summary by court zone
    zones = {"flag-end third": lambda x, y: x < X_MIN + (X_MAX - X_MIN) / 3,
             "centre third": lambda x, y: X_MIN + (X_MAX - X_MIN) / 3 <= x <= X_MAX - (X_MAX - X_MIN) / 3,
             "lounge-end third": lambda x, y: x > X_MAX - (X_MAX - X_MIN) / 3}
    side = lambda x, y: Y_HALF - abs(y) < 1.5
    base = lambda x, y: min(x - X_MIN, X_MAX - x) < 1.5
    inner = lambda x, y: not side(x, y) and not base(x, y)
    print("\nper player-minute            occupancy  new   lost  relink  dup-share")
    for zn, zf in list(zones.items()) + [("within 1.5 m of a sideline", side), ("within 1.5 m of a baseline", base),
                                         ("interior (not near a line)", inner)]:
        m = np.array([zf(x, y) for x, y in occ_xy])
        om = m.sum() / FPS / 60
        ev = collections.Counter(e[0] for e in events if zf(e[2], e[3]))
        dm = sum(zf(x, y) for x, y in dup_xy) / FPS / 60
        print(f"  {zn:27s} {om:7.1f}   {ev['new'] / om:5.2f} {ev['lost'] / om:5.2f} {ev['relink'] / om:6.2f}   {dm / om:6.1%}")


if __name__ == "__main__":
    main()
