### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 52.4% | 38%-61% | -28.2% | 0 / 6 | 30 |
| default (identity output) | 80.6% | 71%-90% | — | — | 26 |
| previous default (no identity freshness) | 80.7% | 74%-88% | +0.1% | 3 / 3 | 26 |
| track IDs instead of identities | 71.6% | 60%-82% | -9.0% | 1 / 5 | 22 |
| no per-player look | 77.4% | 65%-90% | -3.2% | 0 / 5 | 28 |
| old association (isotropic, plain Mahalanobis) | 68.8% | 56%-80% | -11.8% | 0 / 6 | 43 |
| numbers: old rule (team optional, 60 s) | 75.4% | 68%-80% | -5.2% | 0 / 4 | 34 |
| no team colour | 59.9% | 49%-66% | -20.7% | 0 / 6 | 28 |
| no jersey numbers (track IDs) | 64.2% | 59%-67% | -16.4% | 0 / 6 | 28 |
| close-encounter swap resolution | 79.7% | 72%-90% | -0.9% | 2 / 2 | 27 |
| unknown identity after close encounters | 64.8% | 58%-70% | -15.9% | 0 / 6 | 56 |
| majority swap split without recent-read identity | 79.2% | 68%-88% | -1.5% | 1 / 4 | 26 |
| recent-read identity without majority split | 80.5% | 72%-89% | -0.1% | 3 / 2 | 28 |
| stricter identity rule (3 reads, 70 %) | 78.5% | 69%-85% | -2.1% | 2 / 4 | 30 |
| court edges: detections to 0.75 m + 5 s coasting | 78.8% | 70%-87% | -1.8% | 0 / 6 | 33 |
| larger cross-camera noise (0.65 / 0.45 m) | 80.5% | 74%-89% | -0.1% | 4 / 2 | 27 |

Per run and minute (default): mount_r1 game_2to3 87%, mount_r1 game_4to5 75%, mount_r2 game_2to3 90%, mount_r2 game_4to5 75%, mount_r3 game_2to3 85%, mount_r3 game_4to5 71%
