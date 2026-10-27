### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 54.8% | 48%-62% | -21.7% | 0 / 6 | 34 |
| default (identity output) | 76.5% | 65%-84% | — | — | 36 |

Per run and minute (default): int8t10_r1 game_2to3 82%, int8t10_r1 game_4to5 76%, int8t10_r2 game_2to3 84%, int8t10_r2 game_4to5 65%, int8t10_r3 game_2to3 82%, int8t10_r3 game_4to5 70%
