### Ground truth game_2to3 (328 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 46% | 36 | 6.7 | 11 | 45 | 31 | 0 |
| court tracker, no cues | 55% | 29 | 4.1 | 10 | 21 | 8 | 0 |
| + team colour | 55% | 29 | 4.1 | 9 | 22 | 9 | 0 |
| + team colour + numbers | 63% | 30 | 4.4 | 12 | 25 | 14 | 0 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 83% | 28 | 3.3 | 9 | 13 | 13 | 0 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 61% | 39 | 4.7 | 14 | 19 | 15 | 1 |
| old association (isotropic, plain Mahalanobis) + team colour | 52% | 48 | 6.6 | 20 | 32 | 23 | 1 |
