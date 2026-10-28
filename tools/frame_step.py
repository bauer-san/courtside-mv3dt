"""Renumber a run on a reduced-frame-rate dataset (make_clips --step N) to 30 fps frame numbers (GitHub issue #5).

Clip frame k shows the moment of 30 fps frame N k, so every recorded frame number is multiplied by N. Afterwards the
court tracker, the online team model and the ground truth (all in 30 fps frames) work unchanged, and their
time-based rules keep their meaning; the tracker's per-detection rules rescale themselves (world_tracker.for_stride).
Run it after the jersey-read conversion and the appearance extraction (which decode the clips by clip frame). The
originals are kept in <run>/clip_frames/; a second call does nothing.

  python -m tools.frame_step <run>
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from tools import ROOT
from tools.appearance import run_dataset

FILES = ("kafka.jsonl.gz", "jersey.npz", "appearance.npz")


def renumber(run: Path) -> int:
    step = json.loads((run_dataset(run) / "manifest.json").read_text()).get("step", 1)
    keep = run / "clip_frames"
    if step == 1 or keep.exists():
        return step
    keep.mkdir()
    for name in FILES:
        if (run / name).exists():
            shutil.move(run / name, keep / name)
    with gzip.open(keep / "kafka.jsonl.gz", "rt") as fi, gzip.open(run / "kafka.jsonl.gz", "wt") as fo:
        for line in fi:
            m = json.loads(line)
            m["id"] = str(int(m["id"]) * step)
            fo.write(json.dumps(m, separators=(",", ":")) + "\n")
    for name in FILES[1:]:
        if (keep / name).exists():
            a = dict(np.load(keep / name))
            a["frame"] = a["frame"] * step
            np.savez_compressed(run / name, **a)
    return step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    a = ap.parse_args()
    print(f"{a.run}: frames x {renumber(ROOT / 'experiments' / a.run)}")


if __name__ == "__main__":
    main()
