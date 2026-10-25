### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 52.3% | 46%-56% | -18.6% | 0 / 6 | 33 |
| default (identity output) | 70.9% | 66%-79% | — | — | 31 |
| swap split by majority of recent reads | 69.2% | 59%-79% | -1.7% | 3 / 3 | 35 |
| swap split by majority + identity from recent reads | 73.1% | 66%-80% | +2.2% | 4 / 1 | 36 |

Per run and minute (default): base_r1 game_2to3 67%, base_r1 game_4to5 69%, base_r2 game_2to3 76%, base_r2 game_4to5 68%, base_r3 game_2to3 79%, base_r3 game_4to5 66%
