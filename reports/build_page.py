"""Build the web page version of the Phase 7 report (reports/phase7_report.html) from phase7_report.md.

The two bar charts are drawn as inline SVG from the result tables (theme-aware, hover tooltips); the court heat map
and the FL calibration photo are published next to the page (figures/...).

  python3 reports/build_page.py
"""

import html
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent


def table(path: Path) -> list[list[str]]:
    """Rows of the first table in a result file (later tables, e.g. the appendix, are ignored)."""
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in path.read_text().split("\n### ")[0].splitlines()
            if ln.startswith("|") and not ln.startswith("|---")]
    return rows[1:]


def pct(s: str) -> float:
    return float(s.rstrip("%"))


def bar_svg(items: list[tuple], xmax: float, unit_note: str, label: str, whiskers: bool) -> str:
    """items: (name, value, lo, hi, highlight). Horizontal bars, one scale, labels left, values right."""
    row, top, left, width = 34, 8, 330, 360
    h = top + row * len(items) + 34
    sx = lambda v: left + v / xmax * width
    out = [f'<svg class="chart" viewBox="0 0 760 {h}" role="img" aria-label="{html.escape(label)}">']
    for t in range(0, int(xmax) + 1, 20 if xmax > 30 else 2):
        x = sx(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{h - 28}"/>'
                   f'<text class="tick" x="{x:.1f}" y="{h - 12}" text-anchor="middle">{t}</text>')
    for i, (name, v, lo, hi, hl) in enumerate(items):
        y = top + i * row
        tip = f"{name}: {v:.1f}%" + (f" (range {lo:.0f}-{hi:.0f}%)" if whiskers else "")
        out.append(f'<g class="row{" hl" if hl else ""}"><title>{html.escape(tip)}</title>'
                   f'<rect class="hit" x="0" y="{y}" width="760" height="{row}"/>'
                   f'<text class="lab" x="{left - 12}" y="{y + row / 2 + 4}" text-anchor="end">{html.escape(name)}</text>'
                   f'<rect class="bar" x="{left}" y="{y + 8}" width="{max(sx(v) - left, 2):.1f}" height="{row - 16}" rx="3"/>')
        end = sx(v)
        if whiskers:
            out.append(f'<line class="whisk" x1="{sx(lo):.1f}" y1="{y + row / 2}" x2="{sx(hi):.1f}" y2="{y + row / 2}"/>')
            end = max(end, sx(hi))
        out.append(f'<text class="val" x="{end + 8:.1f}" y="{y + row / 2 + 4}">{v:.1f}%</text></g>')
    out.append("</svg>")
    return f'<figure class="fig">{"".join(out)}<figcaption>{unit_note}</figcaption></figure>'


def idf1_chart() -> str:
    rows = table(HERE / "multi_eval_final.md")
    items = []
    for r in rows:
        lo, hi = (float(x) for x in re.findall(r"(\d+)%", r[2]))
        items.append((r[0], pct(r[1]), lo, hi, r[0].startswith("default")))
    items.sort(key=lambda x: -x[1])
    return bar_svg(items, 100, "Identity IDF1 (%). Bar: mean of 3 DeepStream runs × 2 labelled minutes. "
                   "Line: lowest to highest single run and minute. Highlighted: the recommended default.",
                   "Identity IDF1 by tracker configuration", True)


def loss_chart() -> str:
    rows = [r for r in table(HERE / "loss_breakdown_final.md") if r[0] != "correct"]
    items = sorted([(r[0], pct(r[1]), 0, 0, True) for r in rows], key=lambda x: -x[1])
    correct = next(pct(r[1]) for r in table(HERE / "loss_breakdown_final.md") if r[0] == "correct")
    return bar_svg(items, 10, "Share of labelled detections not credited to the right player (%), mean of 3 runs × "
                   f"2 minutes. The remaining {correct:.1f} % are correct.", "Where identity IDF1 is still lost", False)


def photo(src: str, alt: str, caption: str) -> str:
    return (f'<figure class="fig plate"><img src="{src}" alt="{html.escape(alt)}" loading="lazy">'
            f'<figcaption>{caption}</figcaption></figure>')


CSS = """
<title>Player Tracking Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column (~68ch) with wide figures; a scoreboard-style header strip carries the headline numbers */
:root {
  --bg: #f6f7f8; --surface: #ffffff; --fg: #15191e; --muted: #59626c; --rule: #dde1e5;
  --accent: #2a78d6; --accent-soft: #e3eefb; --bar: #b4bac1; --plate: #ffffff;
  --display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111417; --surface: #181c20; --fg: #e8ebee; --muted: #a2abb5; --rule: #2b3137;
  --accent: #4a92ec; --accent-soft: #17283c; --bar: #59616a; --plate: #f4f5f6; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111417; --surface: #181c20; --fg: #e8ebee; --muted: #a2abb5; --rule: #2b3137;
  --accent: #4a92ec; --accent-soft: #17283c; --bar: #59616a; --plate: #f4f5f6; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 17px/1.6 var(--body); }
.wrap { max-width: 980px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.doc { min-width: 0; }
.doc > p, .doc > ul, .doc > ol, .doc > h2, .doc > h3 { max-width: 70ch; }
.doc > .tbl { max-width: 920px; }
header.top { display: grid; gap: 18px; margin-bottom: 36px; }
.eyebrow { font: 600 13px/1 var(--body); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
h1 { font: 700 clamp(34px, 6vw, 52px)/1.02 var(--display); letter-spacing: .01em; margin: 0; text-wrap: balance; }
.meta { color: var(--muted); font-size: 15px; }
.board { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 2px; background: var(--rule);
  border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.board div { background: var(--surface); padding: 14px 16px; display: grid; gap: 2px; }
.board b { font: 600 34px/1 var(--display); font-variant-numeric: tabular-nums; }
.board .hl b { color: var(--accent); }
.board span { color: var(--muted); font-size: 14px; line-height: 1.35; }
nav.toc { font-size: 15px; color: var(--muted); border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding-block: 10px; display: flex; flex-wrap: wrap; gap: 4px 18px; }
nav.toc a { color: inherit; text-decoration: none; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--accent); }
h2 { font: 600 30px/1.1 var(--display); letter-spacing: .015em; margin: 2.2em 0 .5em; text-wrap: balance; }
h3 { font: 600 21px/1.2 var(--display); letter-spacing: .02em; margin: 1.6em 0 .4em; }
p, li { color: var(--fg); }
a { color: var(--accent); }
strong { font-weight: 700; }
code { font: 14px var(--mono); background: var(--accent-soft); padding: 1px 5px; border-radius: 3px; }
pre { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 13.5px; line-height: 1.55; }
.tbl { overflow-x: auto; margin: 1em 0 1.4em; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 15px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 600 13px/1.3 var(--body); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
tr:last-child td { border-bottom: 0; }
.fig { margin: 1.6em 0; max-width: 100%; width: min(920px, 100%); }
.fig figcaption { color: var(--muted); font-size: 14px; margin-top: 8px; max-width: 70ch; }
.plate img { display: block; width: 100%; height: auto; background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; }
svg.chart { width: 100%; height: auto; display: block; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
svg.chart text { font-family: var(--body); font-size: 14px; fill: var(--fg); }
svg.chart .tick { fill: var(--muted); font-size: 12px; }
svg.chart .grid { stroke: var(--rule); stroke-width: 1; }
svg.chart .bar { fill: var(--bar); }
svg.chart .hl .bar { fill: var(--accent); }
svg.chart .whisk { stroke: var(--fg); stroke-width: 2; stroke-linecap: round; opacity: .7; }
svg.chart .val { font-family: var(--mono); font-size: 13px; }
svg.chart .hit { fill: transparent; }
svg.chart .row:hover .hit { fill: var(--accent-soft); }
@media (max-width: 560px) { body { font-size: 16px; } svg.chart text { font-size: 15px; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
</style>
"""


def normalise(md: str) -> str:
    """python-markdown needs a blank line before a list and 4-space indents for nested blocks."""
    out, fence = [], False
    item = re.compile(r"^ *([-*]|\d+\.) ")
    for ln in md.splitlines():
        if ln.startswith("```"):
            fence = not fence
        if not fence:
            m = re.match(r"^( {2,3})(\S.*)$", ln)
            if m:
                ln = "    " + m.group(2)
            if item.match(ln) and out and out[-1].strip() and not item.match(out[-1]):
                out.append("")
        out.append(ln)
    return "\n".join(out)


def main() -> None:
    md = (HERE / "phase7_report.md").read_text()
    md = normalise(md.split("\n", 1)[1])                             # drop the H1: the page header carries it
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
    body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
    figs = {
        "figures/idf1_by_config.png": idf1_chart(),
        "figures/loss_breakdown.png": loss_chart(),
        "figures/court_heatmap.png": photo("figures/court_heatmap.png", "Court heat maps of track events",
                                           "Whole game, s_int8t10_r1, default tracker (team colour + jersey numbers). Rates are per player-minute; "
                                           "panels show occupancy, new identities, lost tracks, relinks, duplicate "
                                           "tracks, camera coverage, ground-truth switches and cross-camera "
                                           "disagreement."),
        "figures/camera_layouts.png": photo("figures/camera_layouts.png", "Simulated camera coverage for six layouts",
                                            "Simulated coverage of six camera layouts: number of cameras that see a "
                                            "standing player at jersey-readable size (at least 150 px). Hatched: two "
                                            "readable views but from directions less than 45 degrees apart. Orange "
                                            "dots: camera positions."),
        "figures/fl_calibration_before_after.jpg": photo("figures/fl_calibration_before_after.jpg",
                                                         "FL camera court overlay before and after re-tracing",
                                                         "FL camera. Top: original calibration with its hand traces "
                                                         "(red). Bottom: lines re-traced from the image; the projected "
                                                         "court (yellow) sits on the paint, and the 3-point arc "
                                                         "(magenta, not used in the fit) lands on the painted arc."),
    }
    for src, rep in figs.items():
        body = re.sub(rf'<p><img alt="[^"]*" src="{re.escape(src)}" ?/?></p>', lambda m, r=rep: r, body)
    toc = re.findall(r'<h2 id="([^"]+)">(.+?)</h2>', body)
    nav = '<nav class="toc" aria-label="Sections">' + "".join(
        f'<a href="#{i}">{re.sub(r"<[^>]+>", "", t)}</a>' for i, t in toc) + "</nav>"
    me = {r[0]: r for r in table(HERE / "multi_eval_final.md")}
    d, mv = me["default (identity output)"], me["MV3DT global IDs"]
    lo, hi = re.findall(r"(\d+)%", d[2])
    header = f"""<header class="top">
<div class="eyebrow">Phase 7 report · 2026-10-27</div>
<h1>Multi-camera basketball player tracking</h1>
<div class="meta">Four cameras, DeepStream 9.1 MV3DT, RTX 3060 Ti; target device Jetson AGX Orin 64 GB or Orin Nano Super.
Time-aligned streams, INT8 detector, cameras calibrated on their physical mounts. Scores are averages over 3 DeepStream runs × 2 hand-labelled game minutes.</div>
<div class="board">
<div class="hl"><b>{d[1]}</b><span>identity IDF1, default pipeline (players reported by team + number)</span></div>
<div><b>{mv[1]}</b><span>identity IDF1, MV3DT global IDs as delivered</span></div>
<div><b>95 fps</b><span>per camera on the 3060 Ti with jersey reading, INT8 (30 needed)</span></div>
<div><b>{lo}-{hi}%</b><span>range of identity IDF1 per run and labelled minute (identical DeepStream runs)</span></div>
</div>
{nav}
</header>"""
    page = CSS + f'<div class="wrap">{header}<main class="doc">{body}</main></div>\n'
    (HERE / "phase7_report.html").write_text(page)
    left = re.findall(r'<img[^>]+src="figures/(idf1|loss)', page)
    print("written", HERE / "phase7_report.html", len(page), "bytes; unreplaced chart images:", left)


if __name__ == "__main__":
    main()
