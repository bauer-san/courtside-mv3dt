### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 34.2% | 29%-41% | -44.5% | 0 / 6 | 52 |
| default (identity output) | 78.7% | 75%-82% | — | — | 34 |

Per run and minute (default): int1_r1 game_2to3 82%, int1_r1 game_4to5 76%, int1_r2 game_2to3 82%, int1_r2 game_4to5 82%, int1_r3 game_2to3 75%, int1_r3 game_4to5 75%
