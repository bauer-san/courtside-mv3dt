### Ground truth game_2to3 (324 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 43% | 34 | 6.6 | 16 | 36 | 51 | 0 |
| court tracker, no cues | 50% | 33 | 4.5 | 10 | 19 | 13 | 0 |
| + team colour | 50% | 33 | 4.8 | 12 | 22 | 16 | 0 |
| + team colour + numbers | 59% | 32 | 5.0 | 11 | 28 | 21 | 0 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 71% | 26 | 3.6 | 9 | 15 | 16 | 0 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 55% | 36 | 5.4 | 16 | 24 | 26 | 0 |
| old association (isotropic, plain Mahalanobis) + team colour | 46% | 46 | 7.3 | 21 | 34 | 37 | 0 |
