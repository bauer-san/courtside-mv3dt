### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 61.5% | 59%-66% | -18.3% | 0 / 6 | 76 |
| default (identity output) | 79.9% | 70%-89% | — | — | 46 |

Per run and minute (default): s_int8t10_r1 game_2to3 83%, s_int8t10_r1 game_4to5 81%, s_int8t10_r2 game_2to3 86%, s_int8t10_r2 game_4to5 70%, s_int8t10_r3 game_2to3 89%, s_int8t10_r3 game_4to5 71%
