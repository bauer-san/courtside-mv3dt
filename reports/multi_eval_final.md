### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 59.6% | 55%-64% | -24.7% | 0 / 6 | 81 |
| default (identity output) | 84.4% | 72%-96% | — | — | 36 |

Per run and minute (default): np_r1 game_2to3 96%, np_r1 game_4to5 74%, np_r2 game_2to3 96%, np_r2 game_4to5 72%, np_r3 game_2to3 94%, np_r3 game_4to5 75%
