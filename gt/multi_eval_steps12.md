### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 52.3% | 46%-56% | -18.6% | 0 / 6 | 33 |
| default (identity output) | 70.9% | 66%-79% | — | — | 31 |
| step 1: identity from recent reads | 72.1% | 66%-79% | +1.2% | 3 / 1 | 35 |
| step 2: doubt after close encounters | 57.3% | 53%-64% | -13.6% | 0 / 6 | 60 |
| steps 1 + 2 | 58.7% | 54%-67% | -12.2% | 0 / 6 | 61 |

Per run and minute (default): base_r1 game_2to3 67%, base_r1 game_4to5 69%, base_r2 game_2to3 76%, base_r2 game_4to5 68%, base_r3 game_2to3 79%, base_r3 game_4to5 66%
