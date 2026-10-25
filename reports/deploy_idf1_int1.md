### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 34.2% | 29%-41% | -36.6% | 0 / 6 | 52 |
| default (identity output) | 70.9% | 56%-80% | — | — | 34 |

Per run and minute (default): int1_r1 game_2to3 56%, int1_r1 game_4to5 72%, int1_r2 game_2to3 77%, int1_r2 game_4to5 80%, int1_r3 game_2to3 67%, int1_r3 game_4to5 74%
