"""Final game-segment evaluation: MV3DT global IDs vs the court tracker with identity cues.

All identity metrics use one fixed set of reference jersey reads per run (see tools/evaluate.identity_metrics),
so variants are comparable. For a DeepStream run with in-pipeline reads (jersey.npz from `jersey from-ds`),
an independent Python extraction (jersey_py.npz) is the reference; otherwise jersey.npz itself (self-referenced:
optimistic for variants that use those same reads).

  python -m tools.final_eval <run> [--frames 1:12600] [--md out.md]
"""

import argparse
import collections
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.evaluate import inside, perf, world_metrics
from tools.jersey import load as load_reads
from tools.stitch import Params as StitchParams, ground_filter, load_cameras, load_detections
from tools.world_tracker import WTParams, track_run

FPS = 30


@dataclass
class _GidTrack:
    """MV3DT global ID as a track (court-filtered, positions averaged over cameras)."""
    out: list = field(default_factory=list)
    dets: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    reads: list = field(default_factory=list)


def mv3dt_tracks(run: Path) -> list:
    d = load_detections(run / "kafka.jsonl.gz")
    keep = ground_filter(d, load_cameras(run / "camInfo_used"), inside, StitchParams(margin=0.3))
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    keys = collections.defaultdict(list)
    for fr, cam, gid, x, y in zip(d["frame"][keep], d["cam"][keep], d["gid"][keep], d["x"][keep], d["y"][keep]):
        per[str(gid)][int(fr)].append((x, y))
        keys[str(gid)].append((int(fr), str(cam), str(gid)))
    out = []
    for gid, frames in per.items():
        t = _GidTrack()
        t.out = [(f, *np.mean(v, axis=0), len(v)) for f, v in sorted(frames.items())]
        t.dets = keys[gid]
        out.append(t)
    return out


def reference(run: Path) -> tuple[dict, str]:
    name = "jersey_py.npz" if (run / "jersey_py.npz").exists() else "jersey.npz"
    j = load_reads(run, name)
    ref = {(int(f), str(c), str(g)): (str(n), float(s))
           for f, c, g, n, s in zip(j["frame"], j["cam"], j["gid"], j["number"], j["score"]) if n}
    return ref, name


COLS = [("variant", "{}"), ("w_ids_per_min", "{:.0f}"), ("w_visible_med", "{:.0f}"), ("w_median_life_s", "{:.1f}"),
        ("id_read_purity", "{:.2f}"), ("id_switches_per_min", "{:.1f}"), ("id_tracks_per_player", "{:.1f}"),
        ("id_coverage", "{:.0%}"), ("w_impure", "{:.2f}")]
HEAD = ["variant", "IDs/min", "visible", "median life s", "number purity", "switches/min", "tracks/player",
        "identified", "team-impure tracks"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--frames", default="1:12600")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_dir() else ROOT / "experiments" / args.run
    f0, f1 = (int(v) for v in args.frames.split(":"))
    ref, ref_name = reference(run)
    num_src = "DeepStream" if (run / "jersey_reads.csv").exists() else "Python"
    variants = [
        ("MV3DT global IDs (court-filtered)", None),
        ("court tracker, no cues", dict(p=WTParams(online_relink=True, use_numbers=False), appearance=False)),
        ("+ team colour", dict(p=WTParams(online_relink=True, use_numbers=False))),
        (f"+ team colour + {num_src} numbers", dict(p=WTParams(online_relink=True))),
        (f"{num_src} numbers only", dict(p=WTParams(online_relink=True), appearance=False)),
    ]
    rows = []
    for name, kw in variants:
        T = mv3dt_tracks(run) if kw is None else track_run(run, **kw)
        m = world_metrics(T, run, frames=(f0, f1), ref_reads=ref)
        m["variant"] = name
        rows.append(m)
        print(" | ".join(fmt.format(m.get(c, float("nan"))) for c, fmt in COLS), flush=True)
    pf = perf(run)
    md = (f"### {run.name} (frames {f0}-{f1}; reference reads: {ref_name}, {len(ref)} reads)\n\n"
          f"Pipeline: {pf.get('fps', float('nan')):.1f} fps per camera, GPU memory peak "
          f"{pf.get('gpu_mem_peak_mib', float('nan')):.0f} MiB. All court-tracker variants are causal (online).\n\n"
          "| " + " | ".join(HEAD) + " |\n|" + "---|" * len(HEAD) + "\n"
          + "".join("| " + " | ".join(fmt.format(r.get(c, float("nan"))) for c, fmt in COLS) + " |\n" for r in rows))
    print(md)
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
