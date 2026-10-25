# Courtside MV3DT

Workspace for running NVIDIA DeepStream Multi-View 3D Tracking (MV3DT) on a 4-camera basketball
court, locally on Ubuntu 24.04 (RTX 3060 Ti, 8 GB, driver 580).

## Cameras and calibration

| Stream | Camera | Mount |
|---|---|---|
| 0 | cam_00 NL | flag-end backboard mount (with FR), looks down at the flag-end half |
| 1 | cam_01 FL | lounge-end backboard mount (with NR), looks across at the flag-end half |
| 2 | cam_02 NR | lounge-end backboard mount (with FL), looks down at the lounge-end half |
| 3 | cam_03 FR | flag-end backboard mount (with NL), looks across at the lounge-end half |

Each pair shares one apparatus on top of a backboard (user, 2026-10-23). The current calibrations put FL 1.1 m and
FR 2.0 m from their partner (further back and higher): a calibration inconsistency, not the physical layout.

World frame: metres, origin centre court, +x toward lounge end, +y toward HEAT-mural sideline, z up.

> **Resolution:** the projection matrices are valid only for **1920x1080 undistorted** pixels.
> Any stage that runs at another resolution needs scaled matrices.

## Setup

One-time host setup (needs sudo):

```bash
./scripts/install_nvidia_container_toolkit.sh
```

The Python venv lives on ext4 at `~/.venvs/mv3dt`, outside this NTFS folder. To use a different
location, set `MV3DT_VENV`.

## Phases

| Phase | Command |
|---|---|
| 1. Workspace scaffold + tests | `./scripts/phase1_setup.sh` |
| 2. 60 s / 180 s synced test clips (from frame 3600) | `./scripts/phase2_clips.sh` |
| 3. DeepStream 9.1 MV3DT environment (repo, container, models, brokers) | `./scripts/phase3_setup.sh` |
| 4. NVIDIA 4-camera sample, as documented (+ GPU log) | `./scripts/phase4_sample.sh` |
| 5a. Court dataset + overlay check (experiments/overlays/) | `./scripts/phase5_dataset.sh` |
| 5b. MV3DT on the court, headless (60 s / 180 s / full) | `./scripts/run_mv3dt.sh courtside_60s c60_pnt` (then `courtside_180s c180_pnt`, `courtside full_pnt`) |

| 6. Variants (detectors, pose, tracker, association, topology, court filter) | `./scripts/phase6_variants.sh courtside_60s c60 <variant ...>`, `./scripts/phase6_tracker_sweep.sh courtside_60s t60 <step ...>` |
| 6b. Court filter app (custom build of deepstream-test5-app) | `./scripts/build_court_app.sh`, then `COURT_FILTER=1 ./scripts/run_mv3dt.sh ...` |
| 7. Evaluation (raw MV3DT, offline clean-up, court-plane tracker) | `python -m tools.evaluate <run ...> --md report.md` |
| 7. Report (draft, 2026-10-23): [reports/phase7_report.md](reports/phase7_report.md); web page `python3 reports/build_page.py` -> reports/phase7_report.html | `python -m tools.multi_eval mount_r1 mount_r2 mount_r3`; figures: `python reports/make_figures.py` |

Offline tracking on top of DeepStream (no GPU, reads `experiments/<run>/kafka.jsonl.gz`):
- `python -m tools.world_tracker <run> --write`: court-plane Kalman tracker over all cameras' foot positions
  (MV3DT IDs ignored), with court filter, per-camera bias correction, duplicate merge, spectator filter and
  stitching. Writes `tracks_world.csv`.
- `python -m tools.render_bev experiments/<run>/tracks_world.csv out.mp4 --start 20 --seconds 20`: BEV clip.

Identity cues for the court tracker (real games only: numbered jerseys, consistent kits; colour cues only during
the game, e.g. `--max-frame 12600` for the first 7 min of the test videos):
- `python -m tools.appearance extract <run>` then `teams <run> --k 5 --max-frame 12600`; check
  `team_clusters.jpg` and set labels with `set-teams <run> 0=maroon 3=red 4=ref ...` (`other` = no team).
- `python -m tools.jersey extract <run> --max-frame 12600`: jersey-number reads (RapidOCR, PP-OCRv4 ONNX).
- `python -m tools.reid extract <run>`: generic ReID embeddings (weak for teammates; not used by the tracker).
- `WTParams(online_relink=True)` is the causal (deployable) mode; offline stitching is for analysis only.
onnxruntime-gpu comes with pip CUDA wheels: call `onnxruntime.preload_dlls()` before creating a session.

Run options for `run_mv3dt.sh`: `DETECTOR_MODEL=PeopleNetTransformer|PeopleNet2.6.3|RTDETR`,
`COURT_FILTER=1` (drop off-court detections before the tracker; margin `COURT_MARGIN`, default 0.75 m),
`NEIGHBORS=all|pairs|pairs_centre` (camera neighbour topology, `tools/pub_sub.py`).
Do not edit `scripts/*.sh` while a run is using them (bash reads scripts incrementally).

`run_mv3dt.sh <dataset> <run> [Section.key=value ...]` writes to `experiments/<run>/`: generated configs,
`outVideos/osd.mp4` (tracking view), `bev/*.mp4` (BEV over `layout_v2.png`), `track_dump/` (per-stream
terminated tracks with foot world X/Y), `tracker-kitti-dump/`, `kafka.jsonl.gz` (all MV3DT messages:
global IDs + per-camera `footLocation3D`), `console.log`, `gpu.csv`, `run.txt`. Detector via
`DETECTOR_MODEL=PeopleNetTransformer|PeopleNet2.6.3|RTDETR`; extra args patch `config_tracker.yml`.

Brokers (after a reboot Kafka must be restarted): `./scripts/brokers.sh start|stop|status`.
MQTT = native mosquitto on :1883 (inter-camera tracker messages); Kafka 4.2.0 in
`external/kafka_2.13-4.2.0` on :9092, topic `mv3dt` (metadata for the BEV visualizer).

DeepStream: `external/DeepStream` at tag `v9.1.0.2`; app in
`src/apps/reference_apps/deepstream-tracker-3d-multi-view`; image
`nvcr.io/nvidia/deepstream:9.1-triton-multiarch`
(`sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`, 27.2 GB).

Clip datasets (`datasets/courtside_60s`, `datasets/courtside_180s`) symlink `camInfo/` (and later
`map.png`, `transforms.yml`) from `datasets/courtside`; each has a `manifest.json` with the source
frame range, decoded frame counts and sync-check results.

Identity ground truth (small, hand-labelled): `python -m tools.gt_prepare full_pn263_cf_jersey` writes
`gt/game_2to3/` (30 moments of 2:00-3:00, every 2 s: each tracked person's camera crops, `tasks.json`,
`sprites/`, and the labelling page `annotate.html`, published as a private claude.ai artifact whose `labels`
collection holds the labels). Save the labels to `gt/game_2to3/labels/` (ArtifactData list with out_dir), then
`python -m tools.gt_eval full_pn263_cf_jersey` scores MV3DT and the court-tracker variants (switches,
tracks per player, impure tracks, duplicates).
A second, held-out set: `python -m tools.gt_prepare full_pn263_cf_jersey --name game_4to5 --start 7200`
(4:00-5:00, more flag-end play); score it with `--name game_4to5`.
DeepStream runs are not repeatable (MV3DT ids/tracklets/reads differ run to run; identity IDF1 moves +-6 pts), so
decisions use several runs: `python -m tools.multi_eval mount_r1 mount_r2 mount_r3` (configurations, paired vs default)
and `python -m tools.loss_breakdown mount_r1 mount_r2 mount_r3` (where identity IDF1 is lost).
Identity output: `identity_tracks(track_run(run, p), track_run.last_det_ident)` reports each detection as the
player (team:number) its track was identified as at that frame (causal; roster inferred online, `track_run.last_roster`).
`python -m tools.gt_causes full_pn263_cf_jersey [--events]` classifies every ground-truth switch (swap,
duplicate, drifted, takeover, lost/reborn) and duplicate (same / different camera).
`python -m tools.xcam full_pn263_cf_jersey` measures cross-camera position disagreement on the labelled
people (pairwise per camera pair; per camera radial = along the line of sight vs tangential error).
`python -m tools.calib_check full_pn263_cf_jersey` refits the cameras on the courtcal line traces (all lines) and with
the far cameras held at their partner's mount; read-only unless `--write-dir` (never overwrites camera files).
`python -m tools.jersey_rec_bench` / `tools.jersey_det_bench` benchmark the DeepStream jersey reader stages offline.
`python -m tools.court_refine <run> <cam>` re-traces a camera's court lines from a median (player-free) image and refits
it. Since 2026-10-23 its result is THE calibration in `datasets/courtside/camInfo` (see camInfo_README.md); runs before
that used the hand-traced originals (each run keeps its own copy in `camInfo_used/`). Test datasets `courtside_full_mount`
(far cameras forced onto their mounts, worse) and `courtside_full_refined` (= the adopted one) were removed.
`python -m tools.render_labels mount_r1` renders the game (0:00-7:00) with the court tracker's player labels: 2x2 cameras +
top-down court, 1280x720 -> experiments/mount_r1/labelled_game.mp4.
`python -m tools.court_heatmap full_pn263_cf_jersey` draws where tracks start, are lost, re-link and duplicate
(per player-minute), camera coverage, the ground-truth switches and cross-camera disagreement on the court.

## Layout

```
inputs/               raw inputs (not in git)
tools/                clip making, validation, evaluation
datasets/courtside/   MV3DT dataset: videos/ (symlinks), camInfo/, map.png, transforms.yml
experiments/          generated configs + run outputs (not in git)
scripts/              run scripts (docker commands)
tests/
```

`datasets/*/videos/` holds **relative** symlinks into `inputs/`. When running in Docker, mount the
whole workspace (not just `datasets/`) so the links resolve.

Videos, model weights and TensorRT engines are never committed (see `.gitignore`).
