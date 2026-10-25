### Identity IDF1 over 3 DeepStream runs x 2 labelled minutes

| configuration | mean IDF1 | spread (min-max) | vs default (paired mean) | pairs better / worse | mean switches |
|---|---|---|---|---|---|
| MV3DT global IDs | 52.3% | 46%-56% | -18.6% | 0 / 6 | 33 |
| default (identity output) | 70.9% | 66%-79% | — | — | 31 |
| track IDs instead of identities | 64.9% | 56%-75% | -6.0% | 0 / 5 | 30 |
| no per-player look | 68.9% | 62%-78% | -2.0% | 2 / 4 | 31 |
| old association (isotropic, plain Mahalanobis) | 63.2% | 53%-69% | -7.7% | 2 / 4 | 42 |
| numbers: old rule (team optional, 60 s) | 65.5% | 57%-75% | -5.4% | 1 / 5 | 40 |
| no team colour | 60.4% | 53%-65% | -10.5% | 0 / 6 | 31 |
| no jersey numbers (track IDs) | 59.8% | 50%-65% | -11.1% | 0 / 6 | 29 |

Per run and minute (default): base_r1 game_2to3 67%, base_r1 game_4to5 69%, base_r2 game_2to3 76%, base_r2 game_4to5 68%, base_r3 game_2to3 79%, base_r3 game_4to5 66%
