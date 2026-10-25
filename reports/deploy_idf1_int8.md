### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 53.4% | 41%-62% | -21.7% | 0 / 6 | 36 |
| default (identity output) | 75.2% | 69%-79% | — | — | 30 |

Per run and minute (default): int8_r1 game_2to3 74%, int8_r1 game_4to5 79%, int8_r2 game_2to3 77%, int8_r2 game_4to5 72%, int8_r3 game_2to3 79%, int8_r3 game_4to5 69%
