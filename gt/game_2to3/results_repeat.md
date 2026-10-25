### Ground truth game_2to3 (326 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 48% | 29 | 6.5 | 12 | 44 | 39 | 0 |
| court tracker, no cues | 51% | 30 | 5.1 | 12 | 25 | 14 | 0 |
| + team colour | 53% | 30 | 5.0 | 13 | 25 | 15 | 0 |
| + team colour + numbers | 59% | 28 | 4.8 | 14 | 27 | 22 | 0 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 69% | 29 | 3.9 | 8 | 18 | 20 | 0 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 46% | 50 | 6.1 | 17 | 24 | 22 | 0 |
| old association (isotropic, plain Mahalanobis) + team colour | 52% | 48 | 7.1 | 16 | 37 | 31 | 0 |
