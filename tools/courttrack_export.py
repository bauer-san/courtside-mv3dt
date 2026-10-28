"""Record the exact per-frame inputs of the court tracker for the C++ port (apps/courttrack), plus the Python
outputs to compare against, and compare a C++ replay with them.

  python -m tools.courttrack_export export mount_r1 OUT_DIR             # offline team model (as in the report)
  python -m tools.courttrack_export export-online mount_r1 OUT_DIR    # main configuration (stored camera bias)
  python -m tools.courttrack_export export-online mount_r1 OUT_DIR --bias-from mount_r2   # bias from another run
  apps/courttrack/ct_replay OUT_DIR/inputs.bin OUT_DIR/cpp_out.txt
  python -m tools.courttrack_export compare mount_r1 OUT_DIR
  python -m tools.courttrack_export live ct_r1 [ct_r2 ...]          # runs with the court tracker in DeepStream

export         inputs as WorldTracker.step receives them inside track_run (online re-linking): ground-filtered,
               camera-bias-corrected floor positions, team labels from the offline team model, heights, jersey
               reads, colour features
export-online  the main configuration (world_tracker.track_default): fixed camera bias (stored in camInfo, or
               estimated on the --bias-from run), no spectator filter, and instead of labels and features the raw torso histograms of good crops;
               the C++ side runs the online team model itself
live           for runs made with COURTTRACK=1 (scripts/run_mv3dt.sh): checks that the C++ replay and the Python
               replay of the probe's recorded inputs both reproduce the live identities, scores the live identities
               on the labelled minutes, and scores the Python reference on the same run's Kafka output with the
               offline (OpenCV) torso histograms, which isolates the effect of computing them in the pipeline
Cameras are numbered in name order (births sort cameras by name).

inputs.bin (little endian):
  "CTIN1" or "CTIN2"  int32 n_cams  n_cams x (float64 x, float64 y, int32 len, name)   camera floor positions
  then per frame: int32 frame, int32 n_cams_in_frame, per camera: int32 cam, int32 n, per detection:
    float64 x, y, team_p, height; int64 gid; int32 len, team; int32 has_read; float64 score; int32 len, number;
    CTIN1: int32 has_feat; [float32 x 144 colour feature]
    CTIN2: int32 has_hist; [float32 x 144 torso histogram]       (team and team_p empty)
  int32 -1
"""

import argparse
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

from tools import ROOT
from tools import world_tracker as wtm
from tools.world_tracker import WTParams, identity_tracks, is_static_edge, track_run


def _s(x) -> bytes:
    b = (x or "").encode()
    return struct.pack("<i", len(b)) + b


class _Writer:
    def __init__(self, path: Path, magic: bytes):
        self.f = open(path, "wb")
        self.magic = magic
        self.cams = None

    def header(self, cam_xy: dict) -> None:
        names = sorted(cam_xy)
        self.cams = {c: i for i, c in enumerate(names)}
        self.f.write(self.magic + struct.pack("<i", len(names)))
        for c in names:
            self.f.write(struct.pack("<dd", *map(float, cam_xy[c])) + _s(c))

    def frame(self, frame, dets, info, reads, extra) -> None:
        """extra(key) -> float32 array (feature or histogram) or None."""
        buf = [struct.pack("<ii", frame, len(dets))]
        for cam, Z in dets.items():
            meta = (info or {}).get(cam)
            buf.append(struct.pack("<ii", self.cams[cam], len(Z)))
            for z, (lab, h, lab_p, key) in zip(Z, meta):
                r = reads.get(key)
                v = extra(key)
                buf.append(struct.pack("<dddd", float(z[0]), float(z[1]), float(lab_p), float(h)))
                buf.append(struct.pack("<q", int(key[2])) + _s(lab))
                buf.append(struct.pack("<i", r is not None) + struct.pack("<d", r[1] if r else 0.0)
                           + _s(r[0] if r else None))
                buf.append(struct.pack("<i", v is not None))
                if v is not None:
                    buf.append(np.asarray(v, dtype="<f4").tobytes())
        self.f.write(b"".join(buf))

    def close(self) -> None:
        self.f.write(struct.pack("<i", -1))
        self.f.close()


def _save_python(out: Path, cams: dict, det_ident: dict, tracks: list) -> None:
    key = lambda k: (int(k[0]), cams[k[1]], int(k[2]))
    with open(out / "python.pkl", "wb") as g:
        pickle.dump({"cams": cams, "det_ident": {key(k): v for k, v in det_ident.items()},
                     "tracks": [(t.tid, [key(k) for k in t.dets], list(t.out)) for t in tracks]}, g)


def export(run: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    w = _Writer(out / "inputs.bin", b"CTIN1")

    class Recorder(wtm.WorldTracker):
        def step(self, frame, dets, info=None):
            if w.cams is None:
                w.header(self.cam_xy)
            w.frame(frame, dets, info, self.reads, self.feats.get)
            super().step(frame, dets, info)

    orig = wtm.WorldTracker
    wtm.WorldTracker = Recorder
    try:
        tracks = track_run(run, WTParams(online_relink=True))
    finally:
        wtm.WorldTracker = orig
    w.close()
    _save_python(out, w.cams, track_run.last_det_ident, tracks)
    print(f"{run.name}: {len(tracks)} tracks, {len(track_run.last_det_ident)} detection identities -> {out}")


def fixed_bias(other: Path) -> dict:
    """Camera bias estimated on another run (stands in for the value measured at installation)."""
    from tools.evaluate import inside
    from tools.stitch import Params as StitchParams
    d = wtm.load_detections(other / "kafka.jsonl.gz")
    keep = wtm.ground_filter(d, wtm.load_cameras(other / "camInfo_used"), inside, StitchParams(margin=0.3))
    return wtm.estimate_bias(wtm._by_frame(d, keep)[0])


def export_online(run: Path, out: Path, bias_from: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    w = _Writer(out / "inputs.bin", b"CTIN2")

    def record(tracker, frame, dets, info, hist):
        if w.cams is None:
            w.header(tracker.cam_xy)
        stripped = {c: [(None, h, 0.0, k) for _, h, _, k in m] for c, m in (info or {}).items()}
        w.frame(frame, dets, stripped, tracker.reads,
                lambda k: None if hist(k) is None else hist(k).astype(np.float32))

    if bias_from is None:
        tracks = wtm.track_default(run, recorder=record)
    else:
        bias = fixed_bias(bias_from)
        orig = wtm.estimate_bias
        wtm.estimate_bias = lambda bf, **kw: bias
        try:
            tracks = wtm.track_default(run, recorder=record, correct_bias=True)
        finally:
            wtm.estimate_bias = orig
    w.close()
    _save_python(out, w.cams, track_run.last_det_ident, tracks)
    print(f"{run.name}: {len(tracks)} tracks, {len(track_run.last_det_ident)} detection identities, "
          f"bias from {bias_from.name if bias_from else 'camInfo'} -> {out}")


def load_cpp(path: Path):
    """cpp_out.txt: 'D frame cam gid identity|-' lines, then per confirmed track 'T tid', 'O frame x y n_cams' and
    'K frame cam gid' lines."""
    det_ident, tracks, cur = {}, [], None
    for line in open(path):
        p = line.split()
        if p[0] == "D":
            det_ident[(int(p[1]), int(p[2]), int(p[3]))] = None if p[4] == "-" else p[4]
        elif p[0] == "T":
            cur = wtm.WTrack(int(p[1]), np.zeros(4), np.eye(4), 0)
            tracks.append(cur)
        elif p[0] == "O":
            cur.out.append((int(p[1]), float(p[2]), float(p[3]), int(p[4])))
        elif p[0] == "K":
            cur.dets.append((int(p[1]), int(p[2]), int(p[3])))
    return det_ident, tracks


def compare(run: Path, out: Path) -> None:
    from tools.gt_eval import load_gt, score
    py = pickle.load(open(out / "python.pkl", "rb"))
    det_ident, tracks = load_cpp(out / "cpp_out.txt")
    names = {i: c for c, i in py["cams"].items()}
    a, b = py["det_ident"], det_ident
    keys = set(a) | set(b)
    same = sum(a.get(k, "?") == b.get(k, "?") for k in keys)
    print(f"detection identities: {same}/{len(keys)} identical ({same / len(keys):.4%}); "
          f"assigned only in Python {len(set(a) - set(b))}, only in C++ {len(set(b) - set(a))}")
    diff = sorted(k for k in keys if a.get(k, "?") != b.get(k, "?"))
    if diff:
        print("first difference at frame", diff[0][0], [(k, a.get(k, "?"), b.get(k, "?")) for k in diff[:5]])
    named = lambda k: (k[0], names[k[1]], str(k[2]))
    py_tracks = []
    for tid, dets, out_ in py["tracks"]:
        t = wtm.WTrack(tid, np.zeros(4), np.eye(4), 0)
        t.dets, t.out = [named(k) for k in dets], out_
        py_tracks.append(t)
    for t in tracks:
        t.dets = [named(k) for k in t.dets]
    # the export stored Python's tracks after its post-processing; apply the same to C++ (offline export only)
    if (out / "inputs.bin").open("rb").read(5) == b"CTIN1":
        tracks = [t for t in tracks if not is_static_edge(t)]
    G_py = identity_tracks(py_tracks, {named(k): v for k, v in a.items()})
    G_cpp = identity_tracks(tracks, {named(k): v for k, v in b.items()})
    for n in ("game_2to3", "game_4to5"):
        gt = load_gt(n, run)
        sp, sc = score(G_py, gt), score(G_cpp, gt)
        print(f"{n}: identity IDF1 Python {sp['idf1']:.2%}  C++ {sc['idf1']:.2%}   "
              f"switches {sp['switches']} / {sc['switches']}")


def live(runs: list[Path]) -> None:
    import csv
    import subprocess
    import tempfile
    from tools.gt_eval import load_gt
    from tools.gt_eval import score as gt_score
    from tools.online_teams import replay_inputs, track_run_online
    N = {n: len(load_gt(n)) for n in ("game_2to3", "game_4to5")}
    rows = []
    for run in runs:
        lv = {(int(r["frame"]), r["sensor"], r["object_id"]): (r["identity"] or None)
              for r in csv.DictReader(open(run / "courttrack_identities.csv"))}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cpp_out.txt"
            subprocess.run([str(ROOT / "apps/courttrack/ct_replay"), str(run / "courttrack_inputs.bin"), str(out)],
                           check=True, capture_output=True)
            cpp, _ = load_cpp(out)
        from tools.online_teams import read_inputs   # camera table: index -> name
        names = {i: n for i, (n, _) in next(read_inputs(run / "courttrack_inputs.bin")).items()}
        cpp = {(k[0], names[k[1]], str(k[2])): v for k, v in cpp.items()}
        T, py = replay_inputs(run / "courttrack_inputs.bin")
        from tools.gt_eval import game_end
        for scope, last in (("game", game_end(run)), ("whole recording", 10 ** 9)):
            kc = [k for k in set(lv) | set(cpp) if k[0] <= last]
            kp = [k for k in set(lv) | set(py) if k[0] <= last]
            same_c = sum(lv.get(k, "?") == cpp.get(k, "?") for k in kc)
            same_p = sum(lv.get(k, "?") == py.get(k, "?") for k in kp)
            print(f"{run.name} ({scope}): live identities {sum(k[0] <= last for k in lv)}; C++ replay identical "
                  f"{same_c}/{len(kc)}, Python replay identical {same_p}/{len(kp)}")
        G_live = identity_tracks(T, lv)
        T_ref = track_run_online(run, correct_bias="stored", post=False)
        G_ref = identity_tracks(T_ref, track_run.last_det_ident)
        for n in N:
            gt = load_gt(n, run)
            a, b = gt_score(G_live, gt), gt_score(G_ref, gt)
            det = len(gt) / N[n]
            rows.append((run.name, n, a["idf1"], b["idf1"], det))
            print(f"  {n}: identity IDF1 live {a['idf1']:.1%} (end-to-end {a['idf1'] * det:.1%}), "
                  f"Python on Kafka + OpenCV histograms {b['idf1']:.1%}; switches {a['switches']} / {b['switches']}")
    if rows:
        m = lambda i: np.mean([r[i] for r in rows])
        print(f"mean over {len(rows)} run-minutes: live {m(2):.1%} (range {min(r[2] for r in rows):.0%}-"
              f"{max(r[2] for r in rows):.0%}), end-to-end {np.mean([r[2] * r[4] for r in rows]):.1%}; "
              f"Python reference {m(3):.1%}; labelled people detected {m(4):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("export", "export-online", "compare", "live"))
    ap.add_argument("run")
    ap.add_argument("out", nargs="*", help="output directory (export, compare); more runs (live)")
    ap.add_argument("--bias-from", help="export-online: run whose camera bias is used instead of the stored one")
    a = ap.parse_args()
    exp = lambda r: Path(r) if Path(r).is_dir() else ROOT / "experiments" / r
    run = exp(a.run)
    if a.cmd == "live":
        live([run] + [exp(r) for r in a.out])
        return
    if len(a.out) != 1:
        ap.error(f"{a.cmd} needs one output directory")
    out = Path(a.out[0])
    if a.cmd == "export":
        export(run, out)
    elif a.cmd == "export-online":
        export_online(run, out, exp(a.bias_from) if a.bias_from else None)
    else:
        compare(run, out)


if __name__ == "__main__":
    sys.exit(main())
