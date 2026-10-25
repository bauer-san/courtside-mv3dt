### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 54.8% | 48%-62% | -19.4% | 0 / 6 | 34 |
| default (identity output) | 74.2% | 61%-83% | — | — | 37 |

Per run and minute (default): int8t10_r1 game_2to3 83%, int8t10_r1 game_4to5 74%, int8t10_r2 game_2to3 83%, int8t10_r2 game_4to5 61%, int8t10_r3 game_2to3 78%, int8t10_r3 game_4to5 67%
