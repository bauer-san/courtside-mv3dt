### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 25.8% | 22%-31% | -51.0% | 0 / 6 | 86 |
| default (identity output) | 76.8% | 63%-89% | — | — | 31 |

Per run and minute (default): int2_r1 game_2to3 89%, int2_r1 game_4to5 63%, int2_r2 game_2to3 87%, int2_r2 game_4to5 66%, int2_r3 game_2to3 85%, int2_r3 game_4to5 70%
