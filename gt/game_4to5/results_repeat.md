### Ground truth game_4to5 (345 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 59% | 32 | 6.4 | 16 | 38 | 47 | 0 |
| court tracker, no cues | 67% | 35 | 4.3 | 13 | 21 | 12 | 1 |
| + team colour | 68% | 34 | 4.1 | 12 | 21 | 14 | 1 |
| + team colour + numbers | 64% | 37 | 4.3 | 10 | 23 | 16 | 1 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 69% | 41 | 4.0 | 11 | 18 | 14 | 1 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 73% | 34 | 4.1 | 11 | 19 | 15 | 1 |
| old association (isotropic, plain Mahalanobis) + team colour | 59% | 43 | 4.8 | 14 | 26 | 20 | 0 |
