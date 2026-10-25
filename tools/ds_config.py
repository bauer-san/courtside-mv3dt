"""Text-level edits of NVIDIA's generated configs (keeps their comments and layout).

Tracker YAML (%YAML:1.0, two-space indented sections):
  python -m tools.ds_config tracker config_tracker.yml TargetManagement.outputTerminatedTracks=1 ...
DeepStream INI (config_deepstream.txt):
  python -m tools.ds_config ini config_deepstream.txt sink2.output-file=outVideos/osd.mp4 ...
"""

import argparse
import re
from pathlib import Path


def set_tracker_key(text: str, section: str, key: str, value: str) -> str:
    lines = text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.rstrip() == f"{section}:")
    except StopIteration:
        return text.rstrip("\n") + f"\n{section}:\n  {key}: {value}\n"
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip() or lines[end].lstrip().startswith("#")):
        end += 1
    for i in range(start + 1, end):
        if re.match(rf"^  {re.escape(key)}:", lines[i]):
            lines[i] = f"  {key}: {value}"
            return "\n".join(lines)
    lines.insert(start + 1, f"  {key}: {value}")
    return "\n".join(lines)


def set_ini_key(text: str, section: str, key: str, value: str) -> str:
    lines = text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == f"[{section}]")
    except StopIteration:
        return text.rstrip("\n") + f"\n\n[{section}]\n{key}={value}\n"
    end = start + 1
    while end < len(lines) and not lines[end].strip().startswith("["):
        end += 1
    for i in range(start + 1, end):
        if lines[i].strip().split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            return "\n".join(lines)
    lines.insert(start + 1, f"{key}={value}")
    return "\n".join(lines)


def apply(path: Path, kind: str, assignments: list[str]) -> None:
    text = path.read_text()
    setter = set_tracker_key if kind == "tracker" else set_ini_key
    for a in assignments:
        lhs, value = a.split("=", 1)
        section, key = lhs.split(".", 1)
        text = setter(text, section, key, value)
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["tracker", "ini"])
    ap.add_argument("path", type=Path)
    ap.add_argument("assignments", nargs="+", help="Section.key=value")
    args = ap.parse_args()
    apply(args.path, args.kind, args.assignments)


if __name__ == "__main__":
    main()
