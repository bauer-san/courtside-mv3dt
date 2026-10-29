# Gym MV3DT

Multi-camera basketball player tracking with NVIDIA DeepStream 9.1 Multi-View 3D Tracking (MV3DT) and a court-plane
tracker that reports each player by identity (team + jersey number). Developed on Ubuntu 24.04 with an RTX 3060 Ti
(8 GB, driver 580); the target is a Jetson (AGX Orin 64 GB or Orin Nano Super), so everything that runs live is
causal. The default configuration: INT8 detector (threshold 0.1), no display, time-aligned camera streams.

Results and the full write-up: [reports/phase7_report.md](reports/phase7_report.md).

![Live output: four camera views with player labels and the top-down court](docs/images/live_output.jpg)

*The court tracker running inside DeepStream (one frame of the live output, 2:45 into the test game): each player
labelled by automatically found team (A/B) and jersey number, and their positions on the court. Faces, the crowd,
signs and the floor logo are pixelated (`python -m tools.render_labels <run> --live --scale 1.5 --sanitize --still
<frame>`).*

![Court heat map: where tracks start, are lost, re-link and duplicate](reports/figures/court_heatmap.png)

*Where on the court tracks start, are lost, re-link and duplicate, per player-minute (`tools/court_heatmap.py`).*

## Cameras and calibration

| Stream | Camera | Mount | View |
|---|---|---|---|
| 0 | cam_00 NL | left backboard (with FR) | looks down at the left half |
| 1 | cam_01 FL | right backboard (with NR) | looks across at the left half |
| 2 | cam_02 NR | right backboard (with FL) | looks down at the right half |
| 3 | cam_03 FR | left backboard (with NL) | looks across at the right half |

Each pair shares one apparatus on top of a backboard, 6-10 in apart. The calibration (one 3 x 4 projection matrix per
camera, valid for **1920 x 1080 undistorted** frames) is fitted with each pair held together on its mount
(`tools/calib_mount.py`); it lives in `datasets/gym/camInfo/`, with earlier versions archived beside it (see
`datasets/gym/camInfo_README.md`). Each camera file also stores the court tracker's fixed position bias
(`courtTrackerBias_m`, measured with `tools/calib_bias.py`).

World frame: metres, origin at centre court, +x toward the right end, +y to the left when facing the right end, z up.
Court size and markings: `tools/court.py`.

**Time alignment:** the four recordings started up to 0.87 s apart. `datasets/gym_sync` holds clips cut so that frame
n of every camera shows the same moment (`tools/make_clips.py --offsets NL=26,FL=13,NR=4`); the offsets were measured
with `tools/sync_check.py` (camera shake from backboard hits, ball-rim contacts). A live system must batch the camera
streams by capture timestamp.

## Setup

- One-time host setup (needs sudo): `./scripts/install_nvidia_container_toolkit.sh`
- Python venv at `~/.venvs/mv3dt` (set `MV3DT_VENV` to use another location)
- Brokers: `./scripts/brokers.sh start|stop|status` (mosquitto on :1883 for MV3DT's inter-camera messages; Kafka
  4.2.0 in `external/kafka_2.13-4.2.0` on :9092). Kafka must be restarted after a reboot.
- DeepStream: `external/DeepStream` at tag `v9.1.0.2`, image `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`.
  The custom app (court filter, torso objects, jersey reads, court tracker) is built with
  `./scripts/build_court_app.sh` into `external/bin/`.

Videos, model weights and TensorRT engines are never committed (see `.gitignore`); `datasets/*/videos/` and
`models/` stay local.

## Jetson AGX Orin

DeepStream 9.1 supports Jetson Orin with **JetPack 7.2 (L4T r39.2)**, and the `deepstream` container repository
serves Jetson and PC from 9.1 on (the image used here has an arm64 variant). MV3DT first appeared in DeepStream 8.0,
so on **JetPack 6** (newest DeepStream: 7.1) the full pipeline can't run.

**JetPack 6 (component measurements, everything in Docker).** On the PC, `./scripts/jetson_bundle.sh <file>.tar
components` (models, the 60 s clips, a live run's recorded court-tracker inputs, ~350 MB). On the device, in a clone
of this repository: `tar -xf <file>.tar`, then `./scripts/agx_components.sh`. It builds a small image on top of one
already on the device (`bauersan/jetson-ndi-yolo`, DeepStream 7.1 for Jetson; else NVIDIA's L4T JetPack image),
adding only what's missing (`trtexec`, `g++`, GStreamer tools), and measures in one container:
- the detector (PeopleNet INT8, batch 4) on the GPU, on DLA 0, on DLA 1 and on both DLAs at once;
- the jersey models;
- the court tracker replayed on recorded inputs (`ct_replay`);
- four 1080p streams through the hardware decoder.

Results land in `experiments/agx_components_<time>/summary.txt`. INT8 scales are placeholders (the PC's calibration
cache is for TensorRT 10.16), which doesn't change the speed. A PC check of what DeepStream 7.1 could offer instead
(MV3DT's multi-view association off, single-view 3D tracking only; `nomv_r1`-`r3`) gave 74.0 % end-to-end against
82.4 % with MV3DT: fewer detections reach the output.

**JetPack 7.2 (full pipeline).** Runs happen on the device; scoring stays on the PC. Steps (GitHub issues #4 and
#6):

1. **Device:** JetPack 7.2 flashed, Docker with the NVIDIA runtime, `mosquitto` running (with
   `set_tcp_nodelay true`, as on the PC), Python 3.12 venv support.
2. **Code and assets:** clone this repository; run `./scripts/phase3_setup.sh` (NVIDIA's DeepStream checkout, Kafka,
   models, container, MV3DT venv) and `./scripts/phase1_setup.sh` (this project's venv). On the PC,
   `./scripts/jetson_bundle.sh <file>.tar` packs what isn't in git (models, time-aligned clips, calibration frames,
   team model); copy it over and `tar -xf` it in the repository root.
3. **Build:** `./scripts/build_court_app.sh` (the CUDA version is taken from the container).
4. **INT8:** first try the PC's calibration cache (third-party JetPack 7.2 guides list TensorRT 10.16, as here): the
   first run builds the engine; check `console.log` for layers falling back to higher precision. If the cache doesn't
   fit, make one on the device ("INT8 on a new device" above) and set `PGIE_INT8_CALIB`.
5. **Runs:** `./scripts/brokers.sh start`, then `./scripts/jetson_bench.sh bias gpu dla live` (camera bias, three
   default runs on the GPU, the detector on each DLA core with `PGIE_DLA_CORE`, and one run with the court tracker
   inside the pipeline). GPU load and shared memory come from `tegrastats` (`tegrastats.log`; there is no
   `nvidia-smi`).
6. **Scoring, on the PC:** copy the `experiments/agx_*` folders back, then
   `python -m tools.calib_bias agx_cal_r1 agx_cal_r2 --into agx_r1 agx_r2 agx_r3 agx_dla0_r1 agx_dla1_r1` (the
   device's camera bias), `python -m tools.multi_eval agx_r1 agx_r2 agx_r3`, `python -m tools.variant_eval --ref
   js_r1 --runs agx_r1 ...` (fps, GPU load, boxes against the PC), and `python -m tools.courttrack_export live
   agx_ct_r1` (live court tracker timing and replay check).

Untested until the device is here: the arm64 container and build, the tegrastats log format, the DLA engines.

## Phases

| Phase | Command |
|---|---|
| 1. Workspace scaffold and tests | `./scripts/phase1_setup.sh` |
| 2. 60 s / 180 s synced test clips | `./scripts/phase2_clips.sh` |
| 3. DeepStream 9.1 MV3DT environment (repo, container, models, brokers) | `./scripts/phase3_setup.sh` |
| 4. NVIDIA's 4-camera sample, as documented | `./scripts/phase4_sample.sh` |
| 5. Court dataset and overlay check | `./scripts/phase5_dataset.sh` |
| 5b. MV3DT on the court, headless | `./scripts/run_mv3dt.sh gym_60s c60_pnt` (also `gym_180s`, `gym_full`) |
| 6. Variants (detectors, pose, tracker, association, topology, court filter) | `./scripts/phase6_variants.sh gym_60s c60 <variant ...>`, `./scripts/phase6_tracker_sweep.sh gym_60s t60 <step ...>` |
| 7. Evaluation and report | `JOBS=8 ./scripts/phase8_reevaluate.sh` (every result table and figure), `python3 reports/build_page.py` (web page) |
| 7b. Detector variants (INT8 thresholds; runs on the aligned clips) | `./scripts/phase7_deploy_variants.sh` |
| 8. Court tracker inside DeepStream | `./scripts/build_court_app.sh`, then `./scripts/phase8_courttrack_live.sh` |

## Running

`./scripts/run_default.sh <run>` runs the default configuration on the time-aligned clips and prepares the run for
scoring. It calls `./scripts/run_mv3dt.sh <dataset> <run> [Section.key=value ...]`, which writes everything to
`experiments/<run>/`: generated configs, `kafka.jsonl.gz` (all MV3DT messages), `console.log`, `gpu.csv`, `run.txt`,
and optionally videos. Options:

| Variable | Effect |
|---|---|
| `DETECTOR_MODEL=PeopleNet2.6.3\|PeopleNetTransformer\|RTDETR` | detector (PeopleNet 2.6.3 is the one used) |
| `COURT_FILTER=1` | drop detections outside the court (+ `COURT_MARGIN`, default 0.75 m) before the tracker |
| `JERSEY=1` | jersey-number reading inside DeepStream (text detector + digit recogniser on torso crops) |
| `COURTTRACK=1` | court tracker + automatic team colours in the pipeline; one Kafka message per frame on topic `courttrack` (needs `JERSEY=1`) |
| `DISPLAY_OUT=0` | no on-screen display or annotated video |
| `PGIE_PRECISION=int8`, `PGIE_INTERVAL=n`, `PGIE_THRESHOLD=x` | detector precision, interval and threshold |
| `PGIE_INT8_CALIB=<file>` | INT8 calibration cache in the model folder (default: the one made for TensorRT 10.16) |
| `PGIE_TOPK=n`, `PGIE_NMS_IOU=x` | boxes kept per frame after NMS (default 20) and its IoU threshold (default 0.5) |
| `NEIGHBORS=all\|pairs\|pairs_centre` | MV3DT camera neighbour topology |

Don't edit `scripts/*.sh` while a run is using them (bash reads scripts as it goes).

### INT8 on a new device

An INT8 calibration cache only works with the TensorRT version it was made with: with NVIDIA's published cache
(TensorRT 10.3), most layers silently fell back to full precision. On every new device or container version (a
Jetson included):
1. Make a cache with that device's TensorRT, from the same calibration frames
   (`experiments/int8_calib/calib_frames.npy`; `tools/int8_calibrate.py` shows the command).
2. Put it in the model folder, set `PGIE_INT8_CALIB` to its name, and delete the cached INT8 engine
   (`resnet34_peoplenet.onnx_b4_gpu0_int8.engine`) so nvinfer rebuilds it.
3. Re-measure the court tracker's camera bias with two unscored runs (`tools/calib_bias.py --write`), since box
   placement changes with the detector build.
4. Re-score accuracy over three runs (`tools/multi_eval.py`).

## Court tracker

- **Python reference:** `tools/world_tracker.py`. `track_default(run)` is the default configuration, causal as
  deployed: automatic team colours (`tools/online_teams.py`), the camera bias stored in camInfo, no offline filters.
  Identity output via `identity_tracks`.
- **C++ port, used in DeepStream:** `apps/courttrack/` (`courttrack.cpp`, `online_teams.cpp`, probe
  `courttrack_probe.cpp`). `make -C apps/courttrack` builds `ct_replay`, which replays recorded inputs; the C++ and
  Python versions give identical output on the same inputs (`tools/courttrack_export.py`).

## Evaluation

Hand-labelled identity ground truth for two minutes of the game is in `gt/game_2to3/` and `gt/game_4to5/` (labels in
`labels/`; the labelling pages are private claude.ai artifacts). DeepStream runs are not repeatable, so results are
averaged over several runs:

- `python -m tools.multi_eval js_r1 js_r2 js_r3 [--jobs 8]`: identity accuracy per run and minute
- `python -m tools.loss_breakdown ...`: where identity accuracy is lost
- `python -m tools.sync_eval <runs>`: aligned vs unaligned clips (MV3DT and court tracker, cross-camera distances)
- `python -m tools.team_label_check <runs>`: the online team labels crop by crop over the whole game
- `python -m tools.nonplayer_eval ...`: spectators and other non-players in the output
- `python -m tools.courttrack_export live <runs>`: runs made with `COURTTRACK=1` (replay checks and scores)
- `python -m tools.render_labels <run> [--live] [--scale 1.5]`: labelled game video (720p, or 1080p with `--scale 1.5`)

## Layout

```
apps/        DeepStream additions: court filter, torso objects, jersey reader parser, court tracker (C++)
datasets/    gym datasets: camInfo/ (calibration), map.png, transforms.yml, videos/ (local only)
experiments/ run outputs (local only)
gt/          identity ground truth
reports/     report, result tables and figures
scripts/     run scripts (Docker commands)
tests/
tools/       Python tools (tracking, evaluation, calibration, rendering)
```

When running in Docker, mount the whole workspace (not just `datasets/`) so relative links resolve.
