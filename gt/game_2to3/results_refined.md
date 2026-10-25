### Ground truth game_2to3 (325 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 50% | 39 | 7.2 | 18 | 41 | 37 | 0 |
| court tracker, no cues | 54% | 36 | 5.2 | 13 | 25 | 18 | 0 |
| + team colour | 50% | 32 | 4.6 | 13 | 23 | 17 | 0 |
| + team colour + numbers | 60% | 32 | 5.1 | 12 | 28 | 25 | 0 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 67% | 34 | 4.7 | 10 | 20 | 22 | 0 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 55% | 45 | 5.7 | 16 | 25 | 23 | 0 |
| old association (isotropic, plain Mahalanobis) + team colour | 52% | 41 | 6.6 | 17 | 33 | 32 | 0 |
