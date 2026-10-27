### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 53.4% | 41%-62% | -22.9% | 0 / 6 | 36 |
| default (identity output) | 76.3% | 68%-85% | — | — | 32 |

Per run and minute (default): int8_r1 game_2to3 85%, int8_r1 game_4to5 74%, int8_r2 game_2to3 77%, int8_r2 game_4to5 71%, int8_r3 game_2to3 83%, int8_r3 game_4to5 68%
