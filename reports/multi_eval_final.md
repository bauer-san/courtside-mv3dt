### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 61.2% | 53%-72% | -24.1% | 0 / 6 | 61 |
| default (identity output) | 85.3% | 74%-96% | — | — | 31 |

Per run and minute (default): js_r1 game_2to3 95%, js_r1 game_4to5 74%, js_r2 game_2to3 93%, js_r2 game_4to5 80%, js_r3 game_2to3 96%, js_r3 game_4to5 74%
