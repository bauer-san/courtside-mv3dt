### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 53.4% | 41%-62% | -23.7% | 0 / 6 | 36 |
| default (identity output) | 77.1% | 69%-82% | — | — | 34 |

Per run and minute (default): int8_r1 game_2to3 82%, int8_r1 game_4to5 69%, int8_r2 game_2to3 78%, int8_r2 game_4to5 77%, int8_r3 game_2to3 79%, int8_r3 game_4to5 78%
