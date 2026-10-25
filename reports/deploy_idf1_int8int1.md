### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 34.4% | 31%-39% | -39.2% | 0 / 6 | 57 |
| default (identity output) | 73.6% | 70%-77% | — | — | 34 |

Per run and minute (default): int8int1_r1 game_2to3 75%, int8int1_r1 game_4to5 74%, int8int1_r2 game_2to3 70%, int8int1_r2 game_4to5 71%, int8int1_r3 game_2to3 77%, int8int1_r3 game_4to5 74%
