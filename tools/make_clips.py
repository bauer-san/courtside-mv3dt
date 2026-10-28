"""Phase 2: cut frame-identical windows from all four camera videos.

Frames are selected by decoded frame number (ffmpeg trim=start_frame/end_frame), not by time,
so the four clips stay in sync. Output: datasets/<name>/videos/cam_XX_POS.mp4, re-encoded
libx264 CRF 18, 30 fps CFR, 1920x1080, no audio. camInfo (and map.png / transforms.yml when
present) are symlinked from the base dataset, so all variants share one calibration.

Verification per clip:
  - exact decoded frame count
  - sync: clip frame 0 must match source frame `start` better than source frames start-1/start+1
A manifest.json with the results is written next to videos/.

--offsets shifts single cameras' windows to correct a time offset between the recordings (GitHub issue #3): with
NL=26, NL's clip starts at source frame start + 26. A camera that shows a moment k frames *later* than another needs
the k-frame later start, so that frame n of every clip shows the same moment. The offsets are recorded in the
manifest; tools.gt_eval shifts ground-truth labels accordingly.

--step N keeps every Nth frame of each window and encodes at 30 / N fps (GitHub issue #5: a lower camera frame
rate). Clip frame k shows source frame start + N k, the same moment on every camera; the step is recorded in the
manifest (tools.frame_step maps a run's frames back to 30 fps numbering for scoring).

Example:
  python -m tools.make_clips --start 3600 --seconds 60 --name gym_60s
  python -m tools.make_clips --start 0 --seconds 898 --offsets NL=26,FL=13,NR=4 --name gym_sync
  python -m tools.make_clips --start 0 --seconds 898 --offsets NL=26,FL=13,NR=4 --step 2 --name gym_sync15
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from tools import CALIB_HEIGHT, CALIB_WIDTH, CAMERAS, DATASETS

FPS = 30
SHARED = ["camInfo", "map.png", "transforms.yml"]
# Small grayscale frames are enough to tell neighbouring frames apart.
PROBE_W, PROBE_H = 480, 270


def encode_cmd(src: Path, dst: Path, start: int, count: int, crf: int = 18, step: int = 1) -> list[str]:
    keep = f",select='not(mod(n,{step}))',setpts=N/({FPS}/{step})/TB" if step > 1 else ",setpts=PTS-STARTPTS"
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", f"trim=start_frame={start}:end_frame={start + count}{keep}",
        "-an", "-r", f"{FPS / step:g}", "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    if (s["width"], s["height"]) != (CALIB_WIDTH, CALIB_HEIGHT):
        raise ValueError(f"{path}: {s['width']}x{s['height']} != calibration resolution")
    return int(s["nb_read_frames"])


def grab_frames(path: Path, first: int, n: int) -> np.ndarray:
    """Decode frames [first, first+n) by frame number as small grayscale arrays."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"trim=start_frame={first}:end_frame={first + n},scale={PROBE_W}:{PROBE_H}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(n, PROBE_H, PROBE_W).astype(np.float32)


def sync_check(src: Path, clip: Path, start: int) -> dict:
    """Mean abs diff of clip frame 0 vs source frames start-1 (if any), start, start+1."""
    first = max(start - 1, 0)
    ref = grab_frames(src, first, start + 2 - first)
    c0 = grab_frames(clip, 0, 1)[0]
    d = [float(np.abs(c0 - r).mean()) for r in ref]
    d_prev = d[0] if start > 0 else None
    d_same, d_next = d[-2], d[-1]
    neighbours = [x for x in (d_prev, d_next) if x is not None]
    return {
        "mad_prev": None if d_prev is None else round(d_prev, 3), "mad_same": round(d_same, 3),
        "mad_next": round(d_next, 3), "ok": d_same < min(neighbours),
    }


def link_shared(base: Path, out: Path) -> None:
    for name in SHARED:
        target = base / name
        link = out / name
        if not target.exists():
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(target, out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=3600, help="first source frame (inclusive)")
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--name", required=True, help="output dataset name under datasets/")
    ap.add_argument("--base", type=Path, default=DATASETS / "gym")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--offsets", default="", help="per-camera extra start frames, e.g. FL=13,NR=22,FR=26")
    ap.add_argument("--step", type=int, default=1, help="keep every Nth frame, encoded at 30 / N fps")
    args = ap.parse_args()
    offsets = {k: int(v) for k, v in (x.split("=") for x in args.offsets.split(",") if x)}
    if set(offsets) - {pos for _, pos, _ in CAMERAS} or any(v < 0 for v in offsets.values()):
        raise SystemExit(f"--offsets: camera positions {[p for _, p, _ in CAMERAS]}, values >= 0")

    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    count = round(args.seconds * FPS)
    out = DATASETS / args.name
    (out / "videos").mkdir(parents=True, exist_ok=True)
    link_shared(args.base, out)

    manifest = {"start_frame": args.start, "frames": count, "fps": FPS / args.step, "step": args.step,
                "crf": args.crf, "offsets": offsets, "cameras": {}}
    expected = -(-count // args.step)
    for cam_id, pos, _ in CAMERAS:
        name = f"{cam_id}_{pos}.mp4"
        src = (args.base / "videos" / name).resolve()
        dst = out / "videos" / name
        start = args.start + offsets.get(pos, 0)
        print(f"[{cam_id}] encoding frames {start}..{start + count - 1} -> {dst.relative_to(DATASETS)}", flush=True)
        subprocess.run(encode_cmd(src, dst, start, count, args.crf, args.step), check=True)
        n = count_frames(dst)
        sync = sync_check(src, dst, start)
        print(f"[{cam_id}] frames={n} sync={sync}", flush=True)
        manifest["cameras"][cam_id] = {"source": src.name, "start_frame": start, "frames": n, "sync": sync}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {c["frames"] for c in manifest["cameras"].values()}
    ok = counts == {expected} and all(c["sync"]["ok"] for c in manifest["cameras"].values())
    print(f"{args.name}: {'OK' if ok else 'FAILED'} (frame counts {sorted(counts)}, expected {expected})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
