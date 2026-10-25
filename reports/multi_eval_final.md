### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 52.4% | 38%-61% | -26.8% | 0 / 6 | 30 |
| default (identity output) | 79.2% | 74%-89% | — | — | 28 |
| previous default (no identity freshness) | 76.4% | 69%-83% | -2.8% | 3 / 3 | 28 |
| track IDs instead of identities | 68.3% | 58%-79% | -11.0% | 1 / 5 | 25 |
| no per-player look | 73.4% | 66%-85% | -5.8% | 0 / 6 | 30 |
| old association (isotropic, plain Mahalanobis) | 66.9% | 63%-72% | -12.3% | 0 / 6 | 43 |
| numbers: old rule (team optional, 60 s) | 74.0% | 69%-82% | -5.2% | 0 / 5 | 43 |
| no team colour | 60.1% | 49%-69% | -19.1% | 0 / 6 | 28 |
| no jersey numbers (track IDs) | 62.2% | 55%-70% | -17.0% | 0 / 6 | 29 |
| close-encounter swap resolution | 74.3% | 60%-89% | -4.9% | 1 / 4 | 30 |
| unknown identity after close encounters | 62.6% | 55%-71% | -16.7% | 0 / 6 | 58 |
| majority swap split without recent-read identity | 77.0% | 69%-83% | -2.3% | 1 / 5 | 28 |
| recent-read identity without majority split | 79.5% | 73%-83% | +0.2% | 2 / 3 | 29 |
| stricter identity rule (3 reads, 70 %) | 74.1% | 64%-83% | -5.1% | 2 / 4 | 31 |
| court edges: detections to 0.75 m + 5 s coasting | 79.5% | 70%-88% | +0.3% | 2 / 4 | 33 |
| larger cross-camera noise (0.65 / 0.45 m) | 74.9% | 70%-82% | -4.3% | 2 / 4 | 32 |

Per run and minute (default): mount_r1 game_2to3 82%, mount_r1 game_4to5 80%, mount_r2 game_2to3 89%, mount_r2 game_4to5 76%, mount_r3 game_2to3 74%, mount_r3 game_4to5 74%
