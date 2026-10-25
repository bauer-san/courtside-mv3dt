"""Figures for the Phase 7 report, from the result tables in reports/.

  python reports/make_figures.py
"""

import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, NEUTRAL = "#2a78d6", "#a9a8a2"      # reference palette: series 1; recessive gray for the other bars


def table(path: Path) -> list[list[str]]:
    """Rows of the first table in a result file (later tables, e.g. the appendix, are ignored)."""
    rows = []
    text = path.read_text().split("\n### ")[0]
    for line in text.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows[1:]                                     # without the header


def pct(s: str) -> float:
    return float(s.rstrip("%"))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def idf1_by_config() -> None:
    rows = table(HERE / "multi_eval_final.md")
    names = [r[0] for r in rows]
    mean = [pct(r[1]) for r in rows]
    lo_hi = [tuple(float(x) for x in re.findall(r"(\d+)%", r[2])) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: mean[i])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.4), facecolor=SURFACE)
    style(ax)
    for y, i in enumerate(order):
        is_default = names[i].startswith("default")
        ax.barh(y, mean[i], height=0.55, color=ACCENT if is_default else NEUTRAL, zorder=2)
        lo, hi = lo_hi[i]
        ax.plot([lo, hi], [y, y], color=INK2, linewidth=2, zorder=3, solid_capstyle="round")
        ax.text(hi + 1, y, f"{mean[i]:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("identity IDF1 (%)  —  bar: mean of 3 DeepStream runs x 2 labelled minutes; line: min-max",
                  color=INK2, fontsize=9)
    ax.set_title("Identity IDF1 by tracker configuration", loc="left", color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "idf1_by_config.png", dpi=150, facecolor=SURFACE)


def loss_breakdown() -> None:
    rows = [r for r in table(HERE / "loss_breakdown_final.md") if r[0] != "correct"]
    rows = sorted(rows, key=lambda r: pct(r[1]))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.4), facecolor=SURFACE)
    style(ax)
    for y, r in enumerate(rows):
        ax.barh(y, pct(r[1]), height=0.55, color=ACCENT, zorder=2)
        ax.text(pct(r[1]) + 0.2, y, f"{pct(r[1]):.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=9)
    ax.set_xlabel("% of labelled detections not credited to the right player (mean, 3 runs x 2 minutes)",
                  color=INK2, fontsize=9)
    ax.set_title("Where identity IDF1 is still lost", loc="left", color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "loss_breakdown.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    idf1_by_config()
    loss_breakdown()
    src = HERE.parent / "experiments" / "calib_refine" / "FL_compare.jpg"
    if src.exists():
        shutil.copy(src, FIG / "fl_calibration_before_after.jpg")
    print(sorted(p.name for p in FIG.iterdir()))
