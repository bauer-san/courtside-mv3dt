"""Interactive interface matrix (N-squared chart) of the default live pipeline, embedded in section 3 of the report
page by build_page.py at the marker <!-- interface-matrix --> in phase7_report.md.

Functions sit on the diagonal in pipeline order; the cell in row i, column j is what function i sends to function j
(above the diagonal: forward, below: feedback). Colour = timing (causal live / fixed before the game / offline here),
diagonal chip = precision. Only functions that run in the default configuration are shown.
"""

import json

NODES = [
    ("align", "Camera time alignment", "gap", [("setupc", "offline")],
     "The four recordings started up to 0.87 s apart. Offsets were measured from camera shake and ball-rim contacts "
     "(issue #3), and the clips were cut so that frame n of every camera shows the same moment (section 2.4).",
     {"Runs": "before the pipeline, on files", "Live": "needs synchronised clocks and capture-time pairing (not built)"}),
    ("cal", "Camera calibration", "setup", [("setupc", "fixed")],
     "One 3 × 4 projection matrix per camera, in metres with the origin at centre court, fitted with the cameras on "
     "their mounts (section 5).",
     {"Runs": "once at installation", "Files": "camInfo/*.yml"}),
    ("bias", "Camera position bias", "setup", [("setupc", "fixed")],
     "A constant floor-position offset per camera, measured from detections of the same person in two cameras "
     "(tools/calib_bias.py, runs npcal_r1 and r2). Re-measured after any change to the calibration, the detector or "
     "MV3DT's body model.",
     {"Runs": "once at installation", "Values": "NL −0.25, NR +0.25, FL +0.07, FR −0.07 m along the court"}),
    ("cache", "INT8 calibration cache", "setup", [("setupc", "fixed")],
     "TensorRT INT8 calibration for PeopleNet, made with the deployment device's TensorRT version (10.16 here) by "
     "tools/int8_calibrate.py.",
     {"Runs": "once per device and TensorRT version"}),
    ("src", "Camera input (decode + muxer)", "live", [("none", "no model")],
     "Four 1920 × 1080, 30 fps streams, hardware-decoded and grouped into one batch per frame: frame n of each camera.",
     {"Runs": "every frame", "Batch": "4 cameras"}),
    ("det", "Person detector (PeopleNet 2.6.3)", "live", [("int8", "INT8")],
     "Detects people in all four frames of a batch at 960 × 544. Threshold 0.1; NMS at IoU 0.5 keeping at most 20 "
     "boxes per camera frame.",
     {"Runs": "every frame, GPU", "Precision": "INT8 (TensorRT)"}),
    ("filt", "Court filter", "live", [("none", "no model")],
     "Probe after the detector. Projects each box's bottom-centre onto the floor and drops boxes outside the court "
     "plus 0.75 m (benches, crowd, scorer's table).",
     {"Runs": "every frame, CPU", "Code": "apps/court_filter/court_filter.c"}),
    ("mv3dt", "MV3DT tracker", "live", [("none", "no model")],
     "NvDCF per-camera tracking with a fixed-height 3D body model, plus multi-view association between the four "
     "camera instances over MQTT. No neural network runs here.",
     {"Runs": "every frame", "Between cameras": "MQTT messages"}),
    ("torso", "Torso objects", "live", [("none", "no model")],
     "Probe after the tracker. For player boxes at least 150 px tall and not cut by the frame edge, every 7th such "
     "frame per player, adds a torso child object: rows 12-62 % of the player's box as it would look upright. The "
     "tilt comes from the camera matrix at MV3DT's foot point (section 4.9).",
     {"Runs": "every frame, CPU", "Code": "apps/court_filter/torso_objects.c"}),
    ("updet", "Upright torso crop (nvdspreprocess)", "live", [("none", "no model")],
     "Copies each torso's bounding area into a GPU buffer and rotates the torso upright into the text detector's "
     "384 × 480 input (letterboxed). Torsos tilted less than 5° are not rotated.",
     {"Runs": "per torso object, GPU (CUDA kernel)", "Code": "apps/jersey_upright"}),
    ("tdet", "Text detector (PP-OCRv4 det)", "live", [("fp16", "FP16")],
     "Finds text boxes in the upright torso crops. In INT8 it gave 14.5 % fewer digit reads (on axis-aligned crops), "
     "so it stays FP16 (section 8).",
     {"Runs": "per torso object, GPU", "Precision": "FP16 (INT8 tested, not adopted)"}),
    ("uprec", "Upright text crop (nvdspreprocess)", "live", [("none", "no model")],
     "A probe turns the detector's text boxes (upright-torso pixels) into rotated rectangles in the frame; this step "
     "rotates each one upright into the recogniser's 48 × 64 input.",
     {"Runs": "per text box, GPU (CUDA kernel)", "Code": "apps/jersey_upright"}),
    ("trec", "Digit recogniser (PP-OCRv4 rec)", "live", [("fp16", "FP16")],
     "Reads 1-2 digit numbers from each upright text box. In INT8 it lost 65 % of the reads (on axis-aligned crops), "
     "so it stays FP16 (section 8).",
     {"Runs": "per text box, GPU", "Precision": "FP16 (INT8 tested, not adopted)"}),
    ("jlog", "Jersey read log", "live", [("none", "no model")],
     "Probe after the jersey reader. Passes every read to the court tracker, which keeps the most confident one per "
     "object and frame, and writes jersey_reads.csv.",
     {"Runs": "every frame, CPU", "Code": "apps/court_filter/jersey_log.c"}),
    ("ct", "Court tracker (C++)", "live", [("none", "no model")],
     "Court-plane Kalman tracker over MV3DT's foot points. Removes the camera bias, drops seated spectators, matches "
     "with team colour and jersey numbers, and assigns identities (team + number) from an online roster. Its output "
     "is final as soon as it's made.",
     {"Runs": "every frame, CPU", "Code": "apps/courttrack"}),
    ("team", "Online team model", "live", [("none", "no model")],
     "Every 5 s, clusters the colour summaries of the last 3 minutes of tracks into two teams and everyone else. The "
     "first model comes after 30 s; the teams are the clusters whose tracks get the most jersey reads. No operator.",
     {"Runs": "every 5 s, CPU", "Code": "apps/courttrack/online_teams.cpp"}),
    ("out", "Kafka output", "live", [("none", "no model")],
     "Message broker for the results: MV3DT's metadata, and one court-tracker message per frame (section 4.5).",
     {"Runs": "every frame", "Topics": "MV3DT metadata, courttrack"}),
]

# (from, to, timing, label, detail, feedback)
EDGES = [
    ("align", "src", "gap", "Aligned clips", "Clips cut with the measured offsets so that frame n matches across "
     "cameras. A live system needs synchronised clocks and capture-time pairing in the muxer; not built.", False),
    ("cal", "filt", "setup", "Projection matrices", "Floor point of each box's bottom-centre, tested against the "
     "court rectangle.", False),
    ("cal", "mv3dt", "setup", "camInfo matrices", "3D foot positions and multi-view association between cameras.", False),
    ("cal", "torso", "setup", "Projection matrices", "Image tilt of the body at each player's foot point, and the "
     "upright box (from $COURT_FILTER_CONFIG).", False),
    ("cal", "ct", "setup", "Projection matrices", "Court positions and the seated-person check (box bottom against "
     "foot point).", False),
    ("bias", "ct", "setup", "Bias per camera", "courtTrackerBias_m from each camInfo file, subtracted from that "
     "camera's foot points before tracking.", False),
    ("cache", "det", "setup", "INT8 cache", "TensorRT builds the INT8 engine from the ONNX model and this cache on "
     "first use.", False),
    ("src", "det", "live", "Frame batch", "Frame n of all four cameras, 1920 × 1080, scaled to 960 × 544 for the "
     "detector.", False),
    ("src", "ct", "live", "Frames (colour)", "Torso colour histograms from good crops (boxes ≥ 80 px) of FL, FR and "
     "NR, every 3rd frame.", False),
    ("det", "filt", "live", "Person boxes", "Boxes with confidence ≥ 0.1, at most 20 per camera frame after NMS.", False),
    ("filt", "mv3dt", "live", "On-court boxes", "Only boxes whose floor point lies within the court plus 0.75 m.", False),
    ("mv3dt", "torso", "live", "Tracked boxes + foot points", "Boxes with per-camera object IDs and MV3DT's world "
     "foot location (for the tilt).", False),
    ("mv3dt", "ct", "live", "Foot points + IDs", "Per-camera foot positions on the floor (m), object IDs and "
     "visibility; box heights from FL and FR for body height.", False),
    ("mv3dt", "out", "live", "MV3DT metadata", "MV3DT's tracking metadata per frame.", False),
    ("torso", "updet", "live", "Torso objects", "Bounding area of each rotated torso, plus its centre, size and "
     "angle; each player every 7th eligible frame.", False),
    ("updet", "tdet", "live", "Upright torso tensors", "384 × 480 inputs, as tensor meta.", False),
    ("tdet", "uprec", "live", "Text boxes", "Text regions in upright-torso pixels, turned into rotated rectangles in "
     "the frame.", False),
    ("uprec", "trec", "live", "Upright text tensors", "48 × 64 inputs, as tensor meta.", False),
    ("trec", "jlog", "live", "Digits + confidence", "1-2 digit strings with confidence; empty when text was found but "
     "no digits.", False),
    ("jlog", "ct", "live", "Jersey reads", "Number and score per object and frame (the most confident read is "
     "used).", False),
    ("ct", "team", "live", "Track colours", "Per-track torso colour summaries and jersey reads per detection.", False),
    ("team", "ct", "live", "Team labels", "A, B or no team. The model used for frame f was built from earlier "
     "frames, so this feedback stays causal.", True),
    ("ct", "out", "live", "Identities + positions", "One JSON message per frame: identity (team + number) and court "
     "position of every player.", False),
]

TOKENS_LIGHT = ("--imx-live: #d6efe6; --imx-live-ink: #0f6b4f; --imx-live-edge: #3aa37f; "
                "--imx-setup: #e3e8f4; --imx-setup-ink: #2f4a86; --imx-setup-edge: #6f88c4; "
                "--imx-gap: #fbe6c4; --imx-gap-ink: #8a5300; --imx-gap-edge: #d99a2b; "
                "--imx-int8: #15191e; --imx-int8-ink: #ffffff; --imx-fp16: #c2410c; --imx-none: #eceff2; "
                "--imx-hl: rgba(21,25,30,.06);")
TOKENS_DARK = ("--imx-live: #143b30; --imx-live-ink: #7fdcb8; --imx-live-edge: #3aa37f; "
               "--imx-setup: #1e2a45; --imx-setup-ink: #a9bdf0; --imx-setup-edge: #6f88c4; "
               "--imx-gap: #3d2c10; --imx-gap-ink: #f4c06a; --imx-gap-edge: #d99a2b; "
               "--imx-int8: #e8ebee; --imx-int8-ink: #111417; --imx-fp16: #fb8a4c; --imx-none: #252b31; "
               "--imx-hl: rgba(232,235,238,.06);")

CSS = """<style>
:root { %s }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { %s } }
:root[data-theme="dark"] { %s }
.imx { margin: 1.6em 0; width: min(920px, 100%%); }
.imx .legend { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 12px; font-size: 13.5px; color: var(--muted); }
.imx .lg-group { display: flex; flex-direction: column; gap: 5px; }
.imx .lg-title { font-size: 11.5px; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.imx .lg-items { display: flex; flex-wrap: wrap; gap: 5px 12px; }
.imx .lg { display: inline-flex; align-items: center; gap: 6px; }
.imx .sw { width: 16px; height: 12px; border-radius: 3px; border: 1.5px solid; }
.imx .sw.live { background: var(--imx-live); border-color: var(--imx-live-edge); }
.imx .sw.setup { background: var(--imx-setup); border-color: var(--imx-setup-edge); }
.imx .sw.gap { background: var(--imx-gap); border-color: var(--imx-gap-edge); }
.imx .scroller { overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px; }
.imx .axis { font-size: 12.5px; color: var(--muted); margin: 0 0 6px; }
.imx .matrix { display: grid; gap: 3px; width: max-content; }
.imx .cell { width: 72px; min-height: 60px; border-radius: 4px; font: 500 10.5px/1.25 var(--body); padding: 4px;
  box-sizing: border-box; display: flex; align-items: center; justify-content: center; text-align: center; }
.imx .empty { outline: 1px dashed var(--rule); outline-offset: -1px; }
.imx .hl.empty { background: var(--imx-hl); }
.imx .edge { border: 1.5px solid; cursor: pointer; }
.imx .edge.live { background: var(--imx-live); border-color: var(--imx-live-edge); color: var(--imx-live-ink); }
.imx .edge.setup { background: var(--imx-setup); border-color: var(--imx-setup-edge); color: var(--imx-setup-ink); }
.imx .edge.gap { background: var(--imx-gap); border-color: var(--imx-gap-edge); color: var(--imx-gap-ink); }
.imx .edge.back { border-style: dashed; }
.imx .node { background: var(--bg); border: 1px solid var(--rule); border-left: 5px solid; flex-direction: column; gap: 4px;
  cursor: pointer; justify-content: space-between; align-items: stretch; text-align: left; padding: 5px 4px 4px 5px; }
.imx .node.live { border-left-color: var(--imx-live-edge); }
.imx .node.setup { border-left-color: var(--imx-setup-edge); }
.imx .node.gap { border-left-color: var(--imx-gap-edge); }
.imx .node .nm { font-weight: 700; color: var(--fg); }
.imx .node .ix { font-family: var(--mono); color: var(--muted); font-size: 9.5px; font-weight: 400; }
.imx .edge:hover, .imx .edge:focus-visible, .imx .sel, .imx .node:hover, .imx .node:focus-visible {
  outline: 2px solid var(--fg); outline-offset: 1px; }
.imx .chip { font: 500 10px/1.3 var(--mono); padding: 1px 5px; border-radius: 3px; align-self: flex-start;
  white-space: nowrap; border: 1.5px solid transparent; background: none; }
.imx .chip.int8 { background: var(--imx-int8); color: var(--imx-int8-ink); }
.imx .chip.fp16 { color: var(--imx-fp16); border-color: var(--imx-fp16); }
.imx .chip.none { background: var(--imx-none); color: var(--muted); }
.imx .chip.setupc { color: var(--imx-setup-ink); border-color: var(--imx-setup-edge); }
.imx .panel { margin-top: 10px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 12px 16px; font-size: 15px; min-height: 9em; }
.imx .panel h4 { font: 600 19px/1.2 var(--display); letter-spacing: .02em; margin: 0 0 4px; }
.imx .panel .kind { font-size: 11.5px; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; margin-bottom: 6px; }
.imx .kind.live { color: var(--imx-live-ink); } .imx .kind.setup { color: var(--imx-setup-ink); }
.imx .kind.gap { color: var(--imx-gap-ink); }
.imx .panel p { margin: 0 0 8px; max-width: 75ch; }
.imx .panel dl { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; margin: 0; font-size: 14px; }
.imx .panel dt { color: var(--muted); } .imx .panel dd { margin: 0; }
.imx figcaption { color: var(--muted); font-size: 14px; margin-top: 8px; max-width: 70ch; }
</style>""" % (TOKENS_LIGHT, TOKENS_DARK, TOKENS_DARK)

BODY = """<figure class="imx" id="interface-matrix">
<div class="legend">
<div class="lg-group"><span class="lg-title">Interface timing (colour)</span><div class="lg-items">
<span class="lg"><span class="sw live"></span>Causal, live: the current frame or earlier ones</span>
<span class="lg"><span class="sw setup"></span>Fixed before the game</span>
<span class="lg"><span class="sw gap"></span>Offline in these tests; live version not built</span></div></div>
<div class="lg-group"><span class="lg-title">Precision (diagonal)</span><div class="lg-items">
<span class="lg"><span class="chip int8">INT8</span>neural model</span>
<span class="lg"><span class="chip fp16">FP16</span>neural model, exception</span>
<span class="lg"><span class="chip none">no model</span>code only</span></div></div>
<div class="lg-group"><span class="lg-title">Direction</span><div class="lg-items">
<span class="lg">solid: forward</span><span class="lg">dashed: feedback</span></div></div>
</div>
<div class="scroller" role="region" aria-label="Interface matrix" tabindex="0">
<p class="axis">Row sends → column receives</p><div class="matrix"></div></div>
<div class="panel" aria-live="polite"></div>
<figcaption>Interfaces of the default live pipeline. Functions sit on the diagonal in pipeline order; each other cell
is what the row's function sends to the column's. Select a cell for details.</figcaption>
</figure>"""

JS = """<script>
(() => {
const root = document.getElementById("interface-matrix");
const N = %s, E = %s;
const idx = Object.fromEntries(N.map((n, i) => [n[0], i]));
const KIND = { live: "Causal, live", setup: "Fixed before the game", gap: "Offline in these tests; live version not built" };
const m = root.querySelector(".matrix"), panel = root.querySelector(".panel");
m.style.gridTemplateColumns = `repeat(${N.length}, 72px)`;
const cells = [], at = {};
E.forEach(e => at[idx[e[0]] + "," + idx[e[1]]] = e);
const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
const chips = p => p.map(([k, t]) => `<span class="chip ${k}">${esc(t)}</span>`).join("");
for (let r = 0; r < N.length; r++) for (let c = 0; c < N.length; c++) {
  const el = document.createElement("div");
  el.className = "cell"; el.dataset.r = r; el.dataset.c = c;
  const e = at[r + "," + c];
  if (r === c) {
    el.className += " node " + N[r][2];
    el.innerHTML = `<span><span class="ix">${String(r + 1).padStart(2, "0")}</span> <span class="nm">${esc(N[r][1])}</span></span><span>${chips(N[r][3])}</span>`;
    el.onclick = () => showNode(r, el);
  } else if (e) {
    el.className += " edge " + e[2] + (e[5] ? " back" : "");
    el.textContent = e[3];
    el.onclick = () => showEdge(e, el);
  } else el.className += " empty";
  if (r === c || e) {
    el.tabIndex = 0; el.setAttribute("role", "button");
    el.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); el.click(); } });
  }
  el.addEventListener("mouseenter", () => hl(r, c));
  el.addEventListener("focus", () => hl(r, c));
  m.appendChild(el); cells.push(el);
}
m.addEventListener("mouseleave", () => hl(-1, -1));
function hl(r, c) { cells.forEach(x => x.classList.toggle("hl", +x.dataset.r === r || +x.dataset.c === c)); }
function sel(el) { cells.forEach(x => x.classList.remove("sel")); el.classList.add("sel"); }
function links(out, i) {
  const xs = E.filter(e => idx[out ? e[0] : e[1]] === i).map(e => esc(N[idx[out ? e[1] : e[0]]][1]) + " (" + esc(e[3]) + ")");
  return xs.length ? xs.join("; ") : "none";
}
function showNode(i, el) {
  const n = N[i]; sel(el);
  const meta = Object.entries(n[5]).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  panel.innerHTML = `<div class="kind ${n[2]}">${KIND[n[2]]}</div><h4>${esc(n[1])}</h4><p>${chips(n[3])}</p><p>${esc(n[4])}</p>
    <dl>${meta}<dt>Receives</dt><dd>${links(false, i)}</dd><dt>Sends</dt><dd>${links(true, i)}</dd></dl>`;
}
function showEdge(e, el) {
  sel(el);
  panel.innerHTML = `<div class="kind ${e[2]}">${KIND[e[2]]}${e[5] ? " · feedback" : ""}</div><h4>${esc(e[3])}</h4>
    <dl><dt>From</dt><dd>${esc(N[idx[e[0]]][1])}</dd><dt>To</dt><dd>${esc(N[idx[e[1]]][1])}</dd></dl><p style="margin-top:8px">${esc(e[4])}</p>`;
}
const ct = idx.ct; showNode(ct, cells[ct * N.length + ct]);
})();
</script>"""


def fragment() -> str:
    """The matrix as one HTML block (style, figure, script) for the report page."""
    return CSS + BODY + JS % (json.dumps(NODES, ensure_ascii=False), json.dumps(EDGES, ensure_ascii=False))
