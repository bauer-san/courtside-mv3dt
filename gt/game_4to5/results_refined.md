### Ground truth game_4to5 (346 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 56% | 27 | 5.3 | 9 | 33 | 44 | 1 |
| court tracker, no cues | 59% | 33 | 4.4 | 13 | 23 | 14 | 4 |
| + team colour | 60% | 33 | 4.4 | 14 | 22 | 14 | 3 |
| + team colour + numbers | 58% | 32 | 4.8 | 15 | 23 | 20 | 2 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 69% | 31 | 3.7 | 7 | 17 | 18 | 2 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 56% | 33 | 4.3 | 12 | 17 | 18 | 2 |
| old association (isotropic, plain Mahalanobis) + team colour | 52% | 47 | 5.7 | 17 | 27 | 22 | 4 |
