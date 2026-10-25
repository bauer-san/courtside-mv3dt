# MV3DT camera files from court calibration

Files: one camInfo YAML per camera, using `projectionMatrix_3x4_w2p` (world metres -> pixel,
origin top-left), as described in the DeepStream Gst-nvtracker "Single-View 3D Tracking" docs.
Stream order must match: cam_00 NL, cam_01 FL, cam_02 NR, cam_03 FR.

IMPORTANT: matrices are for UNDISTORTED frames. Feed MV3DT the *_cfr_undist.mp4 files,
or undistort live streams with the same k1 values before inference/tracking.

Person model: height 1.8 m, radius 0.3 m (edit modelInfo for youth leagues).

BEV map (optional): layout_v2.png is 1 px = 1 inch. World (metres) -> layout pixel:
    u = 499.8 + 39.3701 * x
    v = 344.9 - 39.3701 * y
i.e. 3x3 matrix [[39.3701, 0, 499.8], [0, -39.3701, 344.9], [0, 0, 1]].
Put it into transforms.yml following the sample dataset's format.

## 2026-10-23: adopted calibration (single source of truth)

`camInfo/` now holds the calibration from `tools/court_refine.py`: each camera's court lines re-traced from a
median (player-free) image and the camera refitted (focal length, yaw/pitch/roll, position; standard HS/NCAA
markings, court size as measured by courtcal-proto-0.1). Court-line fit RMS: NL 1.6, FL 2.2, NR 1.4, FR 2.2 px
(hand-traced originals: 4.5, 7.7, 4.5, 3.3 px; FL's traces were bent by lettering). Tracking scored the same as a
repeat run with the old calibration. The originals are in `camInfo_archive_courtcal_proto_2026-10-21/` and
`inputs/mv3dt_camInfo.zip`; `tools/setup_workspace.py` keeps this folder unless `--force-caminfo`.
Known open point: the far cameras (FL, FR) fit ~1.1 m behind their backboard mounts (court size not measured).

## 2026-10-24: calibration on the physical mounts (current)

The camera pairs share one apparatus on top of each backboard (FL with NR, FR with NL, 6-10 in apart). The
re-traced calibration above still placed FL and FR ~1 m behind their mounts. `tools/calib_mount.py` refits all four
cameras jointly on the image traces with each far camera held within 0.25 m (per axis) of its partner, standard
HS/NCAA markings, and the court length and width fitted (now 70.3 x 45.1 ft, in `tools/court.py`). Line-fit RMS:
NL 1.75, NR 2.28, FL 3.02, FR 1.88 px (unconstrained: 1.68, 2.05, 2.41, 1.80). Camera centres: NL (-9.41, 0.05,
3.81), FR (-9.66, 0.02, 4.06), NR (9.57, 0.02, 4.05), FL (9.82, 0.19, 4.16) m. Previous files:
`camInfo_archive_retraced_2026-10-23/`. Note: the BEV map image (layout_v2.png) was drawn to the old court size.
