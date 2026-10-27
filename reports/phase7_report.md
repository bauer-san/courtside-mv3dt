# Multi-camera basketball player tracking: Phase 7 report

Four-camera basketball court, NVIDIA DeepStream 9.1 Multi-View 3D Tracking (MV3DT), evaluated on a Dell XPS 8960
(RTX 3060 Ti 8 GB) with deployment on a Jetson (AGX Orin 64 GB or Orin Nano Super) in mind. Date: 2026-10-27.

This report describes the pipeline's default configuration. It is causal, as it runs live: every input is available
while the game is played, and nothing needs an operator. All results come from DeepStream runs on the final
calibration, which places every camera on its physical mount (section 5): three runs (`mount_r1`-`mount_r3`) for the
main results, three more with the court tracker running inside DeepStream (`ct3_r1`-`ct3_r3`, section 4.6). All
court-tracker figures are regenerated with the final code by `scripts/phase8_reevaluate.sh`.

## 1. Summary

**Recommendation.** Use DeepStream for detection, per-camera tracking and jersey reading. Then run the *court tracker*
developed here on its output and report each player by identity (team + jersey number), not by track ID.

| What is reported | Identity IDF1 (3 DeepStream runs x 2 labelled minutes) |
|---|---|
| **Court tracker, player identities** (Python reference) | **80.6 % (71-90 %)** |
| The same, running live inside DeepStream in C++ (3 further runs, section 4.6) | 80.0 % (71-87 %) |

IDF1 here is the share of labelled detections that end up on the output unit matched to the right player. It's
measured on hand-labelled ground truth (section 2.3). For reference, MV3DT's own global IDs, as DeepStream delivers
them, score 52.4 % (38-61 %) on the same ground truth.

- **Speed on the 3060 Ti:** the DeepStream pipeline with jersey reading runs at about 56 fps per camera (real time
  is 30). The court tracker takes about 1.9 ms per frame on one CPU core in Python, and about 0.1 ms in its C++ port,
  which gives identical output (section 4.5).
- **Running live inside DeepStream (section 4.6):** the C++ court tracker and the automatic team model run as part of
  the pipeline and send one Kafka message per frame with every player's identity and court position. The pipeline
  still runs at about 58 fps per camera.
- **No operator:** the two teams are found automatically from torso colour as the game runs and named A and B. From
  2:00 to the end of the game, no crop got the other team's name in any of six runs (section 4.4). Each camera's
  position correction is measured once at installation.
- **Seated spectators are dropped before tracking (section 4.7):** 4 % of their detections reach the output, none
  with an identity, while 99 % of player detections do.
- **Still not good enough for reliable per-player statistics.** About 20 % of player time is credited to the wrong
  player or no player. Most of that is a *wrong* identity (about 15 %), not a missing one (about 4.5 %; section 4.2).
- **Results vary from run to run.** Repeated DeepStream runs of the same configuration give different MV3DT IDs and
  jersey reads. Per-minute identity IDF1 ranges from 71 % to 90 %, so every number here is an average over three
  runs.
- **Measured deployment options (section 6):**
  - **INT8 detector:** 55 % faster than FP16 on the 3060 Ti, at 6-7 points lower end-to-end identity accuracy
    (4-5 points with a lower detection threshold).
  - **Detector every 2nd or 3rd frame:** costs 14-28 points with MV3DT and barely speeds anything up, so it's ruled
    out.
  - **Jetson (estimated from these measurements):** four cameras at 30 fps look borderline to feasible on an AGX
    Orin 64 GB with INT8, and not feasible on an Orin Nano Super.
- **Camera placement matters more than the number of cameras.** Moving the same four cameras from the backboards to
  high corner mounts would raise the share of the court seen by two cameras at jersey-readable size, from clearly
  different directions, from 20 % to about 80 % (simulated). Six high wall cameras would reach about 96 %, but
  probably need more compute than the Orin Nano Super offers (section 7).

## 2. Setup

### 2.1 Hardware and software
- Ubuntu 24.04, RTX 3060 Ti 8 GB, driver 580, Docker with the NVIDIA Container Toolkit.
- DeepStream 9.1: container `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`, repo `NVIDIA/DeepStream` tag
  `v9.1.0.2`, app `deepstream-test5-app` with a small patch (court filter, torso objects, jersey log, court tracker).
- Brokers: mosquitto (MQTT, inter-camera tracker messages) and Kafka 4.2.0 (MV3DT metadata).

### 2.2 Cameras and video
- **Mounting:** two camera pairs, each on one apparatus on top of a backboard, 6-10 in apart. NL + FR sit at the left
  end, NR + FL at the right end. The N cameras look steeply down at the near half; the F cameras look across at the
  far half.
- **Video:** the cameras record 4K; this pipeline uses a 1920 × 1080, 30 fps, undistorted transcode of about 15
  minutes. **0:00-6:29 is a game** (red vs maroon, 2 referees); after that comes a warm-up, which is out of scope.
- **Calibration:** one 3 × 4 projection matrix per camera, in metres with the origin at centre court, fitted with the
  cameras on their mounts (section 5).

### 2.3 Ground truth
Two labelled minutes of the game: 2:00-3:00 and 4:00-5:00. Each has 30 moments, 2 s apart. Every person on court was
labelled at each moment, giving about 380 labelled groups per minute. About 325 and 343 detections per minute carry
a full identity, across about 10 people per minute (players from both teams, including substitutes).

- **Labelling:** done by the user on a web page, with the tracker's guess shown but not counted.
- **Scoring:** labels attach to detection keys (frame, camera, box), so any tracker that uses the same detections can
  be scored. They carry over to new DeepStream runs by box overlap.
- **Refs:** both referees were labelled only as "ref", so referees aren't scored.
- **Limit:** both minutes have been used for development decisions, so there is no untouched held-out data left
  (section 8).

## 3. Pipeline

1. **Detection:** PeopleNet 2.6.3 (FP16, 960 × 544, every frame, batch of 4 cameras).
2. **Court filter** (custom probe): drops detections whose floor point lies outside the court plus 0.75 m (benches,
   crowd, scorer's table).
3. **MV3DT** (NvDCF tracker with the single-view 3D person model, multi-view association over MQTT). Provides
   per-camera tracklets and 3D foot positions.
4. **Jersey reading** inside DeepStream:
   - torso objects are taken from player boxes at least 150 px tall
   - a text detector (PP-OCRv4 det, 384 × 480) finds text boxes
   - a digit recogniser (PP-OCRv4 rec, 48 × 64) reads 1-2 digit numbers

   About 14.5 reads per second over the game; 94.5-94.9 % of reads checked against the ground truth are correct.
5. **Court tracker** (C++ probe inside DeepStream, `apps/courttrack`; Python reference `tools/world_tracker.py`). A
   court-plane Kalman tracker over MV3DT's foot points, fully causal: its output is final as soon as it's made.
   - **camera position bias:** each camera's foot points carry a constant offset (mainly NL −0.20 m and NR +0.18 m
     along the court). It is measured once at installation from detections of the same person in two cameras
     (`tools/calib_bias.py`), stored in each camera's calibration file as `courtTrackerBias_m`, and subtracted before
     tracking. It is stable across runs (NL −0.18 to −0.21 m, NR +0.17 to +0.20 m on three runs) and must be
     re-measured whenever the calibration changes. The stored values were measured on three runs (`ct2_r1`-`ct2_r3`)
     that are not used for any result here.
   - **seated spectators:** detections whose box bottom and MV3DT foot point disagree in the way seated people do
     are dropped before tracking (section 4.7)
   - **matching:** direction-aware measurement noise and a likelihood cost
   - **identity cues:** team colour from the online team model (below) and jersey numbers, where identity = team +
     number
   - **identity output:** an online roster and a per-player appearance model
   - **identity freshness:** identities come from recent reads (10 s), and a track splits when a majority of its
     recent reads show another number
6. **Online team model** (`tools/online_teams.py`, C++ `apps/courttrack/online_teams.cpp`), with no operator:
   - each court track keeps a colour summary (hue x saturation histogram) of its torso crops from FL, FR and NR;
     every 5 s, the summaries of the last 3 minutes are clustered into three groups: two teams and everyone else
   - **first model** (after 30 s): the two clusters whose tracks get the most jersey reads per detection are the
     teams (players' numbers are read, referees' and spectators' hardly ever); before it, no detection has a team
     label
   - **later models** start from the previous centres, and the team names A and B stay with the clusters nearest the
     previous team centres. The read-rate rule decides again only if that match is ambiguous (another cluster within
     twice the squared distance), which didn't happen during the game after the first model in any run here. So identities (A:23)
     stay stable, and a change in who gets read (for example livelier spectators near the end of a game) can't move
     a team name to another cluster.
   - everything outside the two team clusters, referees included, gets no team label
   - per-camera colour correction is learned as the game runs
7. **Output:** one JSON message per frame on Kafka (section 4.6).

## 4. Results

### 4.1 Speed (RTX 3060 Ti)

| Run | fps per camera | GPU use (median) | GPU memory peak |
|---|---|---|---|
| mount_r1 | 56.0 | 87 % | 2.75 GB |
| mount_r2 | 56.6 | 89 % | 2.68 GB |
| mount_r3 | 57.4 | 89 % | 2.60 GB |

These figures include the on-screen display, the tiler and writing an annotated video, none of which a deployment
needs. Without them the same pipeline runs at 61 fps per camera; with an INT8 detector, at 95 fps (section 6.1).
- **Jersey reader, per crop:** text detector 0.36 ms, recogniser 0.16 ms (TensorRT FP16).
- **Court tracker:** 1.9 ms per frame mean, 4.7 ms p99, on one core of an i7-13700 (Python). The C++ port runs at
  0.08-0.11 ms per frame on the same inputs (section 4.5).

### 4.2 Accuracy

| | Result (3 runs x 2 labelled minutes) |
|---|---|
| Identity IDF1 | **80.6 %** (mean); per run and minute 87, 75, 90, 75, 85, 71 % |
| Identity switches | 26 per labelled minute (mean) |

The first labelled minute (2:00-3:00) scores 85-90 %, the second (4:00-5:00) 71-75 %, in every run.

![Where identity IDF1 is lost](figures/loss_breakdown.png)

- **Wrong number carried over from another player (9.4 %):** a track swaps players and keeps the old identity until
  enough new reads arrive. Reads are sparse: a given player is read only every several seconds.
- **Wrong number, other (4.8 %):** misreads, and identities established from too few reads.
- **Unidentified (3.9 %):** almost all of it (3.5 %) on tracks with readable crops and reads that never reached an
  identity.
- **Wrong team with the right number (0.9 %):** on the numbers both teams wear (1 and 3).
- **Missed by the tracker (0.6 %):** small; part of it is the seated-spectator rule (section 4.7). Detection and court
  filtering aren't the bottleneck.

**Run-to-run variation.** Two DeepStream runs of the same configuration share 98.7 % of their boxes, but the MV3DT
IDs, per-camera tracklets and the jersey reads attached to them come out differently, most likely because the order
in which cameras exchange tracker messages varies. Every number here is therefore a mean over three runs.

### 4.3 Where on the court

![Court heat map](figures/court_heatmap.png)

Over the whole game (mount_r1):

| Zone | New identities per player-minute | Tracks lost per player-minute | Time with a duplicate track |
|---|---|---|---|
| Interior | 1.5 | 1.9 | 3.2 % |
| Within 1.5 m of a sideline | 3.5 | 7.0 | 0.5 % |
| Within 1.5 m of a baseline | 6.6 | 8.8 | 0.1 % |
| Left-end third | 2.3 | 3.1 | 4.6 % |
| Right-end third | 1.5 | 2.1 | 1.6 % |

- **Lines:** tracks are lost about four times as often near the lines as in the interior. Players leave the court
  there, and often only one camera sees them.
- **Ends:** the left end is somewhat worse than the right end.

### 4.4 Team labels

The online team model's labels were checked crop by crop over the whole game against reference colour labels (a
colour clustering of the whole game with clusters named by hand, used only for this check; `tools/team_label_check.py`,
`reports/team_label_check.txt`). Six runs: the three mount runs (Python) and the three `ct3` runs (Python reference).

- **Teams:** maroon was A and red B in every run.
- **First minute:** no labels before the first model at 0:30. Between 0:30 and 1:00, 12-23 % of red crops were
  labelled A; between 1:00 and 2:00, up to 29 crops per run.
- **2:00 to the end of the game (6:29):** no crop got the other team's name, and at most 7 referee crops per run got
  a team label.
- **Unchecked:** the model also gives team names to about 5,400-6,500 crops per run that the reference left
  unlabelled, mostly A. It hasn't been checked whether those are players or people beside the court.
- **Assumption:** the first model's read-rate rule assumes players' numbers are read clearly more often than anyone
  else's shirt in the first 30 s of a game. From 2:00 to the end of the game the team clusters got reads on 2.3-4.6 %
  of detections; the third cluster usually much less (median 0.3-2.0 % per run), but in one run up to 4.1 %. That is
  why later models keep the team names on their clusters instead of applying the rule again.

### 4.5 C++ port of the court tracker

`apps/courttrack` is a C++ port of the court tracker (`courttrack.cpp`) and of the online team model
(`online_teams.cpp`), for use inside the DeepStream pipeline. Both were checked by feeding them the exact per-frame
inputs of the Python version (recorded with `tools/courttrack_export.py`, replayed with `apps/courttrack/ct_replay`):

- **Same output:** on mount_r1-r3, all of about 516,000-517,000 detection identities per run are identical to the
  Python version's, as are identity IDF1 and switch counts on both labelled minutes. This needs the colour statistics
  summed in a fixed order, fused multiply-add disabled in the C++ builds (`-ffp-contract=off`, which matters on the
  Jetson's ARM CPU) and NumPy's percentile formula matched exactly.
- **Speed:** 0.08-0.11 ms per frame on one core of the i7, against 1.9 ms for the Python version. The team model's
  re-clustering every 5 s takes up to about 6 ms.

### 4.6 Running inside DeepStream

The court tracker runs live in the pipeline (`scripts/run_mv3dt.sh` with `COURTTRACK=1`; probe
`apps/courttrack/courttrack_probe.cpp`, after the jersey reader):

- **Inputs:** MV3DT's world foot points, the court filter (0.3 m margin), the stored camera bias, the jersey reads of
  the same frame, and torso colour histograms. Every 3rd frame, the torso of each good crop (FL, FR, NR) is cut out of
  the frame with NVIDIA's `NvBufSurfTransform` into a 32 × 48 tile and its hue x saturation histogram computed on the
  CPU. This works the same way on a Jetson.
- **Processing:** once all four cameras of a frame have arrived, the online team model labels the crops, the court
  tracker steps, and the team model learns from the tracker's assignment.
- **Output:** one JSON message per frame on its own Kafka topic (`courttrack`) with, per player: court track id,
  identity (e.g. `A:23`) and team, position and velocity on the court, whether it was seen this frame or predicted,
  how many cameras saw it, and the detections it came from (camera + MV3DT object id). The per-camera messages are
  unchanged. Consumers should key on the track id: a track's identity can change (when first established, or after
  a swap is detected), and a briefly unseen player is kept with a predicted position for up to 2 s.
- **Logs for evaluation:** the identity of every detection (CSV) and the tracker's inputs, which `ct_replay` and the
  Python reference can replay.

**Checked on three full-game runs** (`ct3_r1`-`ct3_r3`, FP16, no display):

| | Result |
|---|---|
| C++ and Python replays of the recorded inputs | identical to the live output over the game (all of about 190,000 detection identities per run) ¹ |
| Identity IDF1, live | **80.0 %** (71-87 %); end-to-end 79.1 % (98.8 % of labelled people detected) |
| Python reference on the same runs (Kafka output, OpenCV histograms) | 79.1 % |
| Pipeline speed with the court tracker | 57.5-58.2 fps per camera, with the evaluation logs written to this PC's hard disk |
| Court tracker + team model, in the pipeline | 1.1 ms mean, about 11 ms p99, 55-63 ms max per frame |
| Torso histograms, in the pipeline | 2.7 ms mean per four-camera batch (every 3rd frame) |

¹ These runs were recorded before the team-name rule of section 3 (names stay with their clusters) was added. During
the game it made no difference; in the warm-up after it, the current code's replays differ from the recorded output
of two runs from 8:05 and 11:17 on.

- **Live vs Python reference:** the same score (80.0 % against 79.1 %; 2 run-minutes better, 4 within half a point).
  The live pipeline computes the torso histograms differently (NV12 frames scaled to small tiles, against OpenCV on
  full-resolution frames); the team model learns from whichever it gets.
- **Duplicate ids:** MV3DT occasionally gives two different people the same object id in one camera and frame
  (70-200 times per game, several metres apart). Everything downstream is keyed by (frame, camera, id), so only the
  first is kept.
- **Speed in the pipeline:** the same computation takes about 0.1 ms per frame when replayed alone, against 1.1 ms in
  the pipeline, presumably because it shares the CPU with DeepStream's threads (not investigated). The slowest frames
  are the team model's re-clustering, which runs on the probe's streaming thread. On a Jetson, this is the part to
  watch: it can be moved to a worker thread with its result applied a fixed number of frames later, which keeps the
  output reproducible.
- **Logging:** writing the evaluation logs to this PC's hard disk stalls for up to 3.6 s at times. The logs are for
  evaluation only and would be off in deployment. The Kafka send is timed together with them (2.4-2.6 ms per frame
  on average, including the stalls).

### 4.7 Seated spectators

The labels include 160 crops of people who aren't players (453 detections in the three mount runs). Almost all of
them are spectators in the front-row seats, seen by FR (414 of the 453), which looks across the court at them
(`tools/nonplayer_eval.py`, `reports/nonplayer_output.md`).

- **The problem:** their boxes end at the line, so the court filter keeps them. MV3DT's body model assumes a standing
  person, so it places their feet about 2 m inside the court (median), among the players, where they could take
  players' identities.
- **The rule:** for a seated person, the box bottom projected onto the floor lands at their seat, at or just outside
  the line, while MV3DT puts the feet 2 m further in. For players the two agree to about 0.3 m. A detection is dropped
  before tracking when its two foot points are more than 1 m apart *and* its box-bottom point is less than 0.6 m
  inside the lines (boxes cut by the bottom image edge are never dropped). On the labelled detections this catches
  96 % of the spectators and 0.5 % of the players. No pose model is needed; both points are already available, in
  Python and in the DeepStream probe.

| Labelled detections | Spectators in the output | Spectators with an identity | Players in the output |
|---|---|---|---|
| Python reference (mount runs) | 4 % | 0 % | 99 % |
| Live in DeepStream (`ct3` runs) | 4 % | 0.2 % (1 detection) | 99 % |

The rule's thresholds were set on this venue's front row and should be checked at other venues.

## 5. Calibration

The final calibration places each camera where it physically is: the pairs on their backboard apparatus.

- **Line tracing:** each camera's court lines were re-traced automatically from a player-free median image
  (`tools/court_refine.py`). This replaced hand-traced lines; on FL those had followed the lettering and the logo
  instead of the paint, and its overlay was visibly rotated (spotted by the user).
- **Joint fit on the mounts** (`tools/calib_mount.py`): all four cameras are fitted together, with each far camera
  held within 0.25 m per axis of its partner (FL with NR, FR with NL) and the standard HS/NCAA markings. The court's
  length and width are fitted.

| Camera | Court-line fit error | Position (x, y, z), m | Distance to partner |
|---|---|---|---|
| NL | 1.75 px | (−9.41, 0.05, 3.81) | — |
| FR | 1.88 px | (−9.66, 0.02, 4.06) | 0.35 m to NL |
| NR | 2.28 px | (9.57, 0.02, 4.05) | — |
| FL | 3.02 px | (9.82, 0.19, 4.16) | 0.33 m to NR |

- **The constraint costs little:** fitted without the mount constraint, the errors are 1.68, 1.80, 2.05 and 2.41 px.
  FL's larger error comes mostly from its far sideline, which runs along the edge of its image.
- **Court size:** fitted at 70.3 × 45.1 ft. All tools read it from `tools/court.py`.
- **Single source of truth:** the camera files are in `datasets/gym/camInfo`, including the court tracker's camera
  bias (section 3). Earlier versions are archived beside them, and each run keeps the copy it used in `camInfo_used/`.
- **Independent check:** the 3-point arc (19.75 ft) wasn't used in the fit, yet it lands on the painted arc (FL shown
  below).

![FL before and after](figures/fl_calibration_before_after.jpg)

**Position consistency between cameras.** For the same labelled person at the same moment, MV3DT foot points from two
cameras differ by a median 0.37 m (18 % of pairs over 0.9 m; 1,114 pairs over three runs). The error runs mostly
along each camera's line of sight. It comes from estimating feet out of boxes (posture, boxes cut by the frame edge),
not from the calibration.

## 6. Deployment options: measured on the 3060 Ti, estimated for Jetson

### 6.1 Measured: INT8, detector interval and no display

Before any of these is proposed for a Jetson, each was run on the 3060 Ti (three full DeepStream runs per variant
unless noted, all without on-screen display or annotated video) and scored on the ground truth. **End-to-end**
counts correctly identified labelled detections over *all* labelled detections, so people a variant fails to detect
count as errors.

| Variant | fps per camera | GPU use | Labelled people detected | End-to-end identity |
|---|---|---|---|---|
| FP16, detector every frame, with display (mount_r1-r3) | 56.7 | 88 % | 98.8 % | 79.7 % (70-89 %) |
| FP16, detector every frame (1 run) | 61.0 | 87 % | 99.0 % | 78.5 % (71-86 %) |
| **INT8, detector every frame** | **94.6** | 88 % | 94.7 % | **72.3 %** (63-83 %) |
| INT8, detector every frame, detection threshold 0.1 instead of 0.2 | 93.2 ¹ | 90 % | 97.9 % | 74.9 % (63-83 %) |
| FP16, detector every 2nd frame | 75.1 | 76 % | 82.9 % | 65.2 % (60-69 %) |
| FP16, detector every 3rd frame | 45.7 | 68 % | 66.6 % | 51.6 % (40-63 %) |
| INT8, detector every 2nd frame | 92.4 | 76 % | 75.4 % | 59.7 % (57-61 %) |

¹ Mean of two runs. The third ran at 65 fps while other CPU-heavy jobs ran on the same PC, so its speed is
not comparable; its accuracy is included.

Full tables: `reports/deploy_end_to_end.md`, `reports/deploy_variants_detection.md`, `reports/deploy_idf1_*.md`.

- **No display:** about 8 % faster (61 vs 57 fps per camera) with identical output. Assumed for every embedded
  configuration below.
- **INT8 detector:**
  - **Speed:** 55 % faster than FP16 (95 vs 61 fps per camera).
  - **Accuracy:** 6-7 points lower end-to-end (72.3 % vs 78.5-79.7 %). INT8 detects about 4 % fewer labelled
    people (players 93.4 % vs 98.7 %, non-players 70.8 % vs 94.4 %; `reports/deploy_int8_by_category.md`).
  - **Lower detection threshold:** lowering it from 0.2 to 0.1 recovers most of the missed people (97.9 % detected,
    players 96.9 %), and part of the accuracy: end-to-end 74.9 % against 72.3 %, with identity IDF1 on detected
    labels about the same (76.5 % vs 76.3 %). So missed detections explain part of INT8's loss, not all of it.
    INT8 boxes also sit differently: only 88-89 % of them match an FP16 box (IoU ≥ 0.5), against 98.6 % between two
    FP16 runs. That suggests box placement, which feeds the floor positions, as a further cause. Not adopted yet; both
    are being investigated (GitHub issue #1).
  - **Calibration:** NVIDIA's published calibration file was made with an older TensorRT (10.3), which names the
    activation layers differently. With it, most layers fell back to full precision and the detector was slower than
    FP16. The runs above use a calibration made with the deployment TensorRT version (10.16) on 400 frames from the
    video (`tools/int8_calibrate.py`). A Jetson needs a calibration made with its own TensorRT version.
- **Detector every 2nd or 3rd frame (DeepStream `interval`): not usable with MV3DT.**
  - **Accuracy:** it costs 14-28 points end-to-end against the three FP16 runs.
  - **Speed:** it barely helps (every 3rd frame is even slower than every frame). GPU use falls while speed doesn't
    rise, so the bottleneck moves to MV3DT's own tracking work.
  - **Output rate:** with MV3DT's visual tracker off, positions come out only on detector frames.
  - **Tracker changes needed:** MV3DT drops a new track the first time it misses a detection, so the runs needed a
    longer early-termination window (3 frames for every 2nd frame, 4 for every 3rd), and the court tracker's
    frame-counting rules had to be scaled to the detection rate.

  NVIDIA's MV3DT configurations always detect on every frame.

### 6.2 Estimated for AGX Orin 64 GB and Orin Nano Super

These are **estimates**: the 3060 Ti measurements scaled by hardware ratios. Nothing has been run on a Jetson.
Device specifications are NVIDIA's published figures as known at the time of writing; check them against current
spec sheets before buying.

| | AGX Orin 64 GB | Orin Nano Super 8 GB |
|---|---|---|
| GPU | 2048 CUDA cores, 64 tensor cores | 1024 CUDA cores, 32 tensor cores |
| Deep-learning accelerators (DLA) | 2 | none |
| AI performance (INT8, sparse, as quoted) | up to 275 TOPS | up to 67 TOPS |
| CPU | 12-core Cortex-A78AE | 6-core Cortex-A78AE |
| Memory / bandwidth | 64 GB, 205 GB/s | 8 GB, 102 GB/s |
| Hardware video encoder | yes | no |
| Hardware decode, H.265 (approximate) | about 7 × 4K30 or 22 × 1080p30 | about 2 × 4K30 or 11 × 1080p30 |
| Power | 15-60 W | 7-25 W |
| Price (developer kit, approximate) | about US$2,000 | about US$250 |

**Assumed relative speed on this workload:** the Orin Nano Super 5-8 times slower than the 3060 Ti, the AGX Orin
64 GB 2.5-3.5 times slower (twice the Nano's GPU cores and memory bandwidth).

| Configuration (detector every frame, no display) | 3060 Ti (measured) | AGX Orin 64 GB (estimate) | Orin Nano Super (estimate) |
|---|---|---|---|
| FP16: fps per camera with 4 cameras | 61 | 17-24 | 8-12 |
| INT8: fps per camera with 4 cameras | 95 | **27-38** | **12-19** |
| INT8: fps per camera with 6 cameras | 63 | 18-25 | 8-13 |
| INT8: GPU share of one 30 fps camera | 7 % | 17-25 % | 35-56 % |

**What this means:**
- **Orin Nano Super:** four cameras at 30 fps don't look feasible, even with INT8. Detecting every 2nd frame was
  ruled out in 6.1. What remains:
  - two cameras
  - or a lower camera frame rate. Running the cameras at 15 fps, with every frame detected, is a different thing
    from a detector interval and **is untested**.
- **AGX Orin 64 GB:** four cameras at 30 fps with INT8 look borderline to feasible.
  - **Six cameras at 30 fps** don't look feasible on the GPU alone.
  - **Its two DLAs** could take the INT8 detector off the GPU (DeepStream supports this), but their throughput for
    this model must be measured.
  - **4K decoding:** four 4K streams for the hybrid jersey reading (section 7.3) are within its decoder, not the
    Orin Nano Super's.
- **Court tracker:** inside the pipeline on the 3060 Ti PC it takes 1.1 ms per frame on average, with p99 about 11 ms
  and occasional frames up to 63 ms (the team model's re-clustering; section 4.6). The Orin's cores are slower, so
  the worst frames could exceed a 33 ms frame period. That doesn't drop frames by itself, but it needs measuring on
  the device; moving the re-clustering to a worker thread is the fix if needed. Finished tracks also need pruning to
  the relink window (memory and time grow over a game).
- **Brokers:** MV3DT exchanges messages over MQTT between its per-camera trackers; on a single device the broker runs
  locally. Kafka is only needed if metadata leaves the device.

**Recommendation.**
- **For four cameras at 30 fps (current or corner layout), choose the AGX Orin 64 GB.**
- **The Orin Nano Super would need fewer cameras or a lower frame rate,** and a 15 fps test should come first.
- **Before buying, measure on the device:**
  - fps per camera (INT8, no display)
  - DLA throughput (AGX Orin)
  - decode capacity
  - accuracy, re-scored with `tools/multi_eval.py` after re-calibrating INT8 for the device's TensorRT

## 7. Camera placement and quantity

The current cameras sit at backboard height and see players mostly from the side. That is the main reason tracks
swap in close encounters (section 4.2). This section compares candidate layouts with a camera-layout simulator
(`tools/camera_layout.py`) and sets them against compute.

### 7.1 Method
- **Model:** each camera is a 1920 × 1080 pinhole at a given position, aim point and lens field of view, over the
  fitted court. For each 0.5 m floor cell and a standing 1.8 m player, it computes:
  - which cameras see the whole player
  - the player's height in pixels: at least **150 px is readable** for the jersey reader, at least 80 px detectable
  - how steeply each camera looks down, and the floor length hidden behind a player ("shadow"; shorter means fewer
    occlusions)
  - the largest angle between two readable cameras' viewing directions. At least **45°** apart is needed to
    triangulate positions and separate players who overlap in one view.
- **Validation:** for the current cameras, real detection boxes measure 0.87-1.02 times the predicted standing height
  (median per camera, mount_r1, game frames; FL lowest). So the model's pixel sizes are realistic, if somewhat
  generous for FL.
- **Fair comparison:** each new camera's lens (50-95° horizontal) and aim point are chosen by a search that maximises
  the area seen by two readable cameras at least 45° apart. Existing cameras stay as calibrated.
- **Venue assumptions, to be replaced by measurements:** walls 2 m beyond the lines, wall mounts up to 6 m high. Room
  dimensions and ceiling height weren't available.

### 7.2 Results

![Camera layouts](figures/camera_layouts.png)

| Layout (mounts 6 m high) | Streams | ≥2 readable views, ≥45° apart | ≥3 readable views | No readable view | Median look-down / shadow |
|---|---|---|---|---|---|
| A. Current: 4 on the backboards | 4 | 20 % | 0 % | 6 % | 15° / 9.2 m |
| E. Keep the 2 far backboard cameras, move 2 to the sideline walls at half court | 4 | 65 % | 35 % | 0 % | 20° / 6.1 m |
| **C. 4 high in the corners** (lenses about 60°, aimed across the court) | **4** | **78 %** | 53 % | 9 % | 26° / 4.4 m |
| B. Current 4 + 2 on the sideline walls at half court | 6 | 70 % | 53 % | 0 % | 30° / 4.0 m |
| D. 4 corners + 2 on the sideline walls at half court | 6 | 93 % | 69 % | 1 % | 27° / 4.3 m |
| **F. 4 on the sideline walls at the quarter lines + 1 per baseline wall** (lenses 50-70°) | **6** | **96 %** | 64 % | 0 % | 28° / 4.1 m |

The table and figure are regenerated by `python -m tools.camera_layout` into `reports/camera_layouts.md` and
`reports/figures/camera_layouts.png`.

**Mounting height** (`reports/camera_layouts_height.md`):

| Height | C: ≥2 readable views ≥45° apart | C: no readable view | C: shadow | F: ≥2 readable views ≥45° apart | F: shadow |
|---|---|---|---|---|---|
| 4.5 m | 85 % | 2 % | 7.5 m | 98 % | 6.3 m |
| 5 m | 84 % | 2 % | 6.4 m | 97 % | 5.2 m |
| 6 m | 78 % | 9 % | 4.4 m | 96 % | 4.1 m |
| 7 m | 80 % | 3 % | 4.2 m | 92 % | 3.1 m |

- **Lower mounts:** players appear larger, so slightly more of the court is readable.
- **Higher mounts:** steeper views, which shrink the area hidden behind each player.
- **For the corner layout, about 5 m is the best balance:** 84 % two-camera coverage with only 2 % of the court
  unreadable.
- **The six-camera layout** works well anywhere from 4.5 to 6 m.

### 7.3 Compute per camera stream

| Platform (INT8 detector every frame, no display) | GPU per 30 fps stream | 4 streams | 6 streams |
|---|---|---|---|
| RTX 3060 Ti | about 7 % (measured) | about 28 % | about 42 % |
| AGX Orin 64 GB | about 17-25 % (estimate) | about 70-100 % | about 105-150 % (not feasible on the GPU alone) |
| Orin Nano Super | about 35-56 % (estimate) | about 140-220 % (not feasible) | not feasible |

- **Basis:** the measured INT8 runs (95 fps per camera at 88 % GPU use, section 6.1). The estimates assume the AGX
  Orin 64 GB is 2.5-3.5 times slower and the Orin Nano Super 5-8 times slower. Neither assumption has been measured.
- **DLAs:** on the AGX Orin, running the detector on its DLAs would free GPU time; not measured.
- **Detector interval:** not an option with MV3DT (section 6.1).
- **Jersey reading** scales with the number of readable players rather than cameras. More cameras can be held to the
  same read budget.
- **Inter-camera tracker messages** (MQTT) grow with the number of camera pairs. Six cameras is modest.
- **The cameras already record 4K;** this pipeline uses a 1080p transcode. The detector runs at 960 × 544 either way.
  So feeding the 4K streams costs mainly decoding (about 4× the decode work, to be checked on the Orin Nano Super),
  while the jersey reader would get torso crops with twice the pixels.
  - **What it wouldn't change:** with the current cameras, the simulator shows readable coverage unchanged (94 % of
    the court by at least one camera). Wherever a camera sees a player at all, the player is already at least 150 px
    tall at 1080p.
  - **What it could change:** read reliability (more reads and fewer misreads on sharp crops), which would shorten
    the wrong-identity periods after swaps (section 4.2).
  - **Lighter variant if decode is the limit:** a 1080p stream for detection and tracking plus a 4K stream at
    5-10 fps for reading only.

### 7.4 Proposal
1. **Same compute as today (4 streams): move to layout C, four corner cameras at about 5 m.**
   - Lenses about 60° horizontal, each aimed diagonally across the court.
   - Two-camera, well-separated readable coverage rises from 20 % to about 84 %.
   - Views become steeper (median 15° → 20°), and the area hidden behind players shrinks from about 9 m to about
     6 m.
   - This is the best accuracy per unit of compute. With four streams it needs an AGX Orin 64 GB at 30 fps
     (section 6.2).
2. **If six streams are affordable, use layout F:** four sideline cameras at the quarter lines plus one per baseline,
   at 5-6 m, for about 96 % coverage. Six streams at 30 fps exceed an AGX Orin's GPU by the estimates in section 6.2,
   unless its DLAs take the detector or the cameras run at a lower frame rate (both untested).
3. **If cameras can't move:** keep the far backboard cameras and move the two near ones to the sideline walls at half
   court (layout E, still 4 streams, 65 %). Adding two cameras to the current four (layout B, 6 streams, 70 %) costs
   more compute and still does worse than four corner cameras.
4. **Whatever the layout:**
   - use identical cameras with locked exposure and focus
   - synchronise frames (hardware sync or accurate timestamps)
   - avoid pointing at windows or bright lights
   - calibrate with the mount positions as a constraint, as in section 5, ideally with measured court dimensions

**Caveat.** These are geometric coverage figures, not tracking accuracy. Before buying hardware, the best check is a
temporary installation (stands or poles at the corners) for one session, run through the same pipeline and scored
on newly labelled ground truth.

**How the professional systems compare** (general knowledge, not verified here):
- **Basketball:** NBA optical tracking (SportVU, Second Spectrum, now Hawk-Eye) uses many fixed cameras high in the
  arena rafters.
- **American football:** the NFL uses RFID chips in the shoulder pads.
- **Identity:** in all of these, operators assign identities, correct swaps live and review data after games, with
  rosters and substitutions from official feeds.
- **What this means here:** high mounts and roster and substitution input are the parts worth borrowing; both fit this
  project's real-time setup.

## 8. Limitations

- **Small ground truth:** two minutes of one game on one court, with 10 players and refs not scored.
- **No clean held-out data:** both labelled minutes informed development decisions, including the online team
  model's settings. A third labelled minute would give an unbiased final number.
- **Run-to-run variation:** per-minute identity IDF1 ranges from 71 % to 90 % across DeepStream runs. Three runs
  narrow it, but differences under about 2 points aren't reliable.
- **Sensitivity:** tracking decisions amplify tiny input changes (a few centimetres of camera bias, a few hundred
  duplicate detections in a game), which shows up as run-to-run spread.
- **One game, one venue:** the online team model, the roster inferred online and the seated-spectator rule's
  thresholds are tested on this one game. Other courts, lighting and kits are untested.
- **Court size and venue:** the court's length and width are fitted, not measured. The camera-placement study assumes
  wall distances and mounting heights.

## 9. Next steps (prioritised)

1. **Check camera synchronisation** (GitHub issue #3): frame timing of the 4K originals, hard sync events and
   cross-camera trajectory correlation, then the position error any offset causes.
2. **Read jerseys from the native 4K frames** (issue #2): detect and track on 1080p as now, cut torso crops from the
   4K frames. Offline experiment first; it targets the largest error (wrong identities after swaps) through more
   reliable reads.
3. **INT8 accuracy** (issue #1): box placement and the lower threshold on the 3060 Ti, then an INT8 profile for the
   Jetson.
4. **Measure on the target Jetson (AGX Orin 64 GB, or Orin Nano Super with fewer cameras or 15 fps).**
   - Re-calibrate INT8 for its TensorRT version (`tools/int8_calibrate.py`).
   - Measure fps per camera without display, and DLA throughput (AGX Orin).
   - Measure the court tracker's worst-frame time in the pipeline; move the team model's re-clustering to a worker
     thread if it exceeds the frame period.
   - Re-score accuracy with `tools/multi_eval.py` and `tools/courttrack_export.py live`.
5. **Test a lower camera frame rate (15 fps) on the 3060 Ti** if the Orin Nano Super stays in consideration. Every
   frame is detected, so MV3DT works as designed; it needs the same three-run evaluation as section 6.1.
6. **Trial the corner layout (C) for one session** with temporary mounts at about 5 m, label a minute and compare with
   the current layout. Simulated two-camera coverage rises from 20 % to about 84 % at the same compute.
7. **More reads per player where it matters.** Re-read tracks for a second or two after a close encounter, and tracks
   without an identity.
8. **Label a third minute** (5:00-6:00 is still unused) for a clean held-out score before any final claims.
9. **Use the official roster** (numbers per team, available for real games): reject reads of numbers on neither
   roster, and settle the team of a number only one team wears. The roster can't say which colour is which team, but
   the tracker can learn that from which numbers each colour cluster is read with. This also targets the wrong-team
   errors on shared numbers.

## 10. How to reproduce

```bash
./scripts/brokers.sh start
for r in mount_r1 mount_r2 mount_r3; do
  DETECTOR_MODEL=PeopleNet2.6.3 COURT_FILTER=1 JERSEY=1 NEIGHBORS=all ./scripts/run_mv3dt.sh gym_full $r
  ~/.venvs/mv3dt/bin/python -m tools.jersey from-ds $r
  cp experiments/full_pn263_cf_jersey/{team_model.npz,teams.yml} experiments/$r/   # reference colour labels (4.4 only)
  ~/.venvs/mv3dt/bin/python -m tools.appearance extract $r
done
./scripts/phase7_deploy_variants.sh   # deployment variants (no display, INT8, detector interval), section 6.1
~/.venvs/mv3dt/bin/python -m tools.variant_eval --ref nodisp_r1 --runs int8_r1 int1_r1 --md reports/deploy_variants_detection.md
~/.venvs/mv3dt/bin/python -m tools.calib_bias ct2_r1 ct2_r2 ct2_r3 --write   # camera bias into camInfo (runs not evaluated)
./scripts/build_court_app.sh && ./scripts/phase8_courttrack_live.sh ct3_r1 ct3_r2 ct3_r3   # section 4.6
JOBS=8 ./scripts/phase8_reevaluate.sh   # every court-tracker table and figure in this report (sections 4-6)
~/.venvs/mv3dt/bin/python -m tools.camera_layout --validate mount_r1 --md reports/camera_layouts.md
python3 reports/build_page.py   # web page version
make -C apps/courttrack && ~/.venvs/mv3dt/bin/python -m tools.courttrack_export export-online mount_r1 /tmp/ct \
  && apps/courttrack/ct_replay /tmp/ct/inputs.bin /tmp/ct/cpp_out.txt \
  && ~/.venvs/mv3dt/bin/python -m tools.courttrack_export compare mount_r1 /tmp/ct   # C++ vs Python, section 4.5
```

- **Calibration:** `tools/court_refine.py <run> <cam>` (line traces), then `tools/calib_mount.py --write`.
- **Ground truth:** `gt/game_2to3`, `gt/game_4to5` (labels in `labels/`; the labelling pages are private claude.ai
  artifacts).
- **Unit tests:** `~/.venvs/mv3dt/bin/python -m pytest -q` (58 pass).
