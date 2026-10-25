"""Recover an H.264 MP4 whose moov index was never written (pipeline stopped without EOS).

The mdat payload is AVCC (4-byte length-prefixed NAL units) with SPS/PPS in-band, as written by
DeepStream's nvv4l2h264enc + qtmux. Converts it to an Annex-B stream and remuxes at a fixed rate.

  python -m tools.recover_mp4 broken.mp4 recovered.mp4 [--fps 30]
"""

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path


def find_mdat(f, size: int) -> tuple[int, int]:
    off = 0
    while off < size:
        f.seek(off)
        h = f.read(16)
        sz, typ = struct.unpack(">I4s", h[:8])
        hdr = 8
        if sz == 1:
            sz, hdr = struct.unpack(">Q", h[8:16])[0], 16
        if sz == 0 or off + sz > size:
            sz = size - off
        if typ == b"mdat":
            return off + hdr, off + sz
        off += sz
    raise ValueError("no mdat box")


def avcc_to_annexb(src: Path, dst: Path) -> int:
    size = src.stat().st_size
    n = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        p, end = find_mdat(f, size)
        f.seek(p)
        while p + 4 <= end:
            (length,) = struct.unpack(">I", f.read(4))
            if length == 0 or p + 4 + length > end:
                break  # truncated tail
            out.write(b"\x00\x00\x00\x01" + f.read(length))
            p += 4 + length
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=args.dst.parent) as tmp:
        raw = Path(tmp) / "stream.h264"
        n = avcc_to_annexb(args.src, raw)
        print(f"{n} NAL units extracted")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-r", str(args.fps), "-i", str(raw),
                        "-c", "copy", "-movflags", "+faststart", str(args.dst)], check=True)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
