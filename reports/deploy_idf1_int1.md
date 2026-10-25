### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 34.2% | 29%-41% | -39.1% | 0 / 6 | 52 |
| default (identity output) | 73.3% | 67%-79% | — | — | 32 |

Per run and minute (default): int1_r1 game_2to3 72%, int1_r1 game_4to5 71%, int1_r2 game_2to3 79%, int1_r2 game_4to5 78%, int1_r3 game_2to3 67%, int1_r3 game_4to5 72%
