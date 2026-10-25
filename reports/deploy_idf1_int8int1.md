### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 34.4% | 31%-39% | -35.4% | 0 / 6 | 57 |
| default (identity output) | 69.8% | 66%-72% | — | — | 36 |

Per run and minute (default): int8int1_r1 game_2to3 66%, int8int1_r1 game_4to5 71%, int8int1_r2 game_2to3 69%, int8int1_r2 game_4to5 68%, int8int1_r3 game_2to3 72%, int8int1_r3 game_4to5 72%
