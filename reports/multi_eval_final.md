### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 52.4% | 38%-61% | -25.3% | 0 / 6 | 30 |
| default (identity output) | 77.8% | 72%-84% | — | — | 33 |
| previous default (no identity freshness) | 75.6% | 68%-82% | -2.1% | 2 / 4 | 31 |
| track IDs instead of identities | 65.8% | 55%-70% | -11.9% | 0 / 6 | 29 |
| no per-player look | 72.3% | 67%-84% | -5.4% | 0 / 5 | 35 |
| old association (isotropic, plain Mahalanobis) | 68.9% | 61%-76% | -8.9% | 0 / 6 | 44 |
| numbers: old rule (team optional, 60 s) | 72.2% | 66%-78% | -5.5% | 0 / 6 | 42 |
| no team colour | 59.8% | 42%-70% | -18.0% | 0 / 6 | 30 |
| no jersey numbers (track IDs) | 62.0% | 54%-68% | -15.7% | 0 / 6 | 29 |
| close-encounter swap resolution | 75.7% | 68%-82% | -2.1% | 2 / 4 | 32 |
| unknown identity after close encounters | 62.6% | 58%-68% | -15.2% | 0 / 6 | 62 |
| majority swap split without recent-read identity | 75.9% | 68%-81% | -1.9% | 1 / 4 | 30 |
| recent-read identity without majority split | 77.1% | 70%-82% | -0.7% | 2 / 3 | 33 |
| stricter identity rule (3 reads, 70 %) | 73.1% | 67%-80% | -4.7% | 0 / 5 | 35 |
| court edges: detections to 0.75 m + 5 s coasting | 77.9% | 71%-88% | +0.2% | 4 / 2 | 31 |
| larger cross-camera noise (0.65 / 0.45 m) | 73.6% | 69%-79% | -4.2% | 1 / 3 | 34 |

Per run and minute (default): mount_r1 game_2to3 81%, mount_r1 game_4to5 73%, mount_r2 game_2to3 84%, mount_r2 game_4to5 78%, mount_r3 game_2to3 79%, mount_r3 game_4to5 72%
