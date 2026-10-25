### Ground truth game_4to5 (339 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 58% | 32 | 7.2 | 18 | 42 | 45 | 1 |
| court tracker, no cues | 57% | 35 | 4.3 | 10 | 17 | 20 | 1 |
| + team colour | 60% | 34 | 4.6 | 13 | 20 | 20 | 1 |
| + team colour + numbers | 63% | 36 | 4.9 | 15 | 21 | 23 | 1 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 64% | 38 | 4.3 | 8 | 15 | 21 | 1 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 66% | 32 | 4.7 | 13 | 18 | 20 | 1 |
| old association (isotropic, plain Mahalanobis) + team colour | 53% | 40 | 6.8 | 18 | 31 | 47 | 1 |
