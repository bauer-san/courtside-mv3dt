"""Phase 5: complete datasets/gym/ in the structure of NVIDIA's sample (datasets/mtmc_4cam):

  camInfo/cam_XX_POS.yml  projectionMatrix_3x4_w2p + modelInfo for classID 0 and 1
                          (person is class 1 for PeopleNetTransformer, class 0 for PeopleNet 2.6.3 / RT-DETR;
                          the detector configs filter out the other classes)
  map.png                 -> inputs/layout_v2.png (symlink)
  transforms.yml          T_ov2px: world (m) -> map pixel; T_gt2px: same (ground truth is in world frame)

Clip datasets (gym_*) get map.png / transforms.yml symlinked from the base dataset.
Run after tools.setup_workspace. Idempotent.
"""

import argparse
import os
from pathlib import Path

from tools import CAMERAS, DATASETS, INPUTS
from tools.court import T_WORLD_TO_MAP

PERSON_CLASS_IDS = (0, 1)


def model_info_block(height: float, radius: float) -> str:
    lines = [
        "# Person model (metres). classID 1 = person for PeopleNetTransformer;",
        "# classID 0 = person for PeopleNet 2.6.3 and RT-DETR. Other classes are filtered by the detector configs.",
        "modelInfo:",
    ]
    for cid in PERSON_CLASS_IDS:
        lines += [f"  - classID: {cid}", f"    height: {height}", f"    radius: {radius}"]
    return "\n".join(lines) + "\n"


def set_model_info(path: Path, height: float, radius: float) -> None:
    """Replace everything from the person-model comment / modelInfo key onward."""
    text = path.read_text()
    cut = text.find("# Person model")
    if cut < 0:
        cut = text.find("modelInfo:")
    head = text if cut < 0 else text[:cut]
    path.write_text(head.rstrip() + "\n\n" + model_info_block(height, radius))


def transforms_yaml(T) -> str:
    def rows(name):
        return f"{name}:\n" + "".join(f"  - {v:.6f}\n" for v in T.ravel())

    return (
        "# World (metres, origin centre court) -> map.png pixel (layout_v2.png, 1 px = 1 inch):\n"
        "#   u = 499.8 + 39.3701 * x,  v = 344.9 - 39.3701 * y\n"
        "# Row-major 3x3, same layout as NVIDIA's datasets/mtmc_4cam/transforms.yml.\n"
        + rows("T_ov2px")
        + "\n# Ground truth (if any) is in the same world frame.\n"
        + rows("T_gt2px")
    )


def relink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASETS / "gym")
    ap.add_argument("--height", type=float, default=1.8, help="person model height (m)")
    ap.add_argument("--radius", type=float, default=0.3, help="person model radius (m)")
    args = ap.parse_args()

    ds = args.dataset
    for cam_id, pos, _ in CAMERAS:
        p = ds / "camInfo" / f"{cam_id}_{pos}.yml"
        set_model_info(p, args.height, args.radius)
        print(f"modelInfo classes {PERSON_CLASS_IDS} h={args.height} r={args.radius}: {p.relative_to(DATASETS)}")
    relink(ds / "map.png", INPUTS / "layout_v2.png")
    (ds / "transforms.yml").write_text(transforms_yaml(T_WORLD_TO_MAP))
    print(f"map.png -> {os.readlink(ds / 'map.png')}; transforms.yml written")

    for clip in sorted(DATASETS.glob(f"{ds.name}_*")):
        for name in ("map.png", "transforms.yml"):
            relink(clip / name, ds / name)
        print(f"linked map.png, transforms.yml into {clip.name}")


if __name__ == "__main__":
    main()
