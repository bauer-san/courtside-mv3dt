"""Phase 1: populate datasets/courtside/ from ./inputs.

- Extracts mv3dt_camInfo.zip into datasets/courtside/camInfo/ as cam_00_NL.yml ... cam_03_FR.yml
- Symlinks (relative, not copies) the undistorted videos into datasets/courtside/videos/
  as cam_00_NL.mp4 ... cam_03_FR.mp4

map.png and transforms.yml are added in Phase 5, after reading the sample dataset's format.
Idempotent: safe to re-run.
"""

import argparse
import os
import zipfile
from pathlib import Path

from tools import CAMERAS, DATASETS, INPUTS


def extract_caminfo(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        for cam_id, pos, _ in CAMERAS:
            name = f"{cam_id}_{pos}.yml"
            if name not in members:
                raise FileNotFoundError(f"{name} missing from {zip_path}")
            dst = out_dir / name
            dst.write_bytes(zf.read(members[name]))
            written.append(dst)
        # Keep the calibration README next to the camInfo files for provenance.
        if "README.md" in members:
            (out_dir.parent / "camInfo_README.md").write_bytes(zf.read(members["README.md"]))
    return written


def link_videos(inputs: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for cam_id, pos, prefix in CAMERAS:
        matches = sorted(inputs.glob(f"{prefix}_*_undist.mp4"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one {prefix}_*_undist.mp4 in {inputs}, got {matches}")
        dst = out_dir / f"{cam_id}_{pos}.mp4"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(os.path.relpath(matches[0], out_dir))
        linked.append(dst)
    return linked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--dataset", type=Path, default=DATASETS / "courtside")
    ap.add_argument("--force-caminfo", action="store_true",
                    help="overwrite datasets/courtside/camInfo with the original calibration zip")
    args = ap.parse_args()

    # the adopted calibration (tools/court_refine.py, see camInfo_README.md) replaced the zip's: keep it
    if (args.dataset / "camInfo").is_dir() and any((args.dataset / "camInfo").glob("cam_*.yml")) \
            and not args.force_caminfo:
        print(f"camInfo  kept existing {args.dataset / 'camInfo'} (--force-caminfo restores the original zip)")
    else:
        for p in extract_caminfo(args.inputs / "mv3dt_camInfo.zip", args.dataset / "camInfo"):
            print(f"camInfo  {p.relative_to(args.dataset)}")
    for p in link_videos(args.inputs, args.dataset / "videos"):
        print(f"video    {p.relative_to(args.dataset)} -> {os.readlink(p)}")


if __name__ == "__main__":
    main()
