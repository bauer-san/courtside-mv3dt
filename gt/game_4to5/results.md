### Ground truth game_4to5 (349 labelled detections, 10 people)

| variant | IDF1 | switches | tracks/player | impure tracks | tracks | duplicates | missed dets |
|---|---|---|---|---|---|---|---|
| MV3DT global IDs (court-filtered) | 55% | 34 | 6.4 | 15 | 35 | 44 | 0 |
| court tracker, no cues | 65% | 32 | 4.0 | 12 | 23 | 11 | 0 |
| + team colour | 65% | 31 | 3.9 | 13 | 22 | 11 | 0 |
| + team colour + numbers | 66% | 29 | 4.5 | 12 | 23 | 14 | 0 |
| + team colour + numbers, IDENTITY OUTPUT (online roster, per-player look) | 72% | 43 | 4.3 | 8 | 18 | 13 | 0 |
| + team colour + numbers, old rule (team optional, 60 s relink) | 63% | 39 | 4.5 | 12 | 19 | 11 | 0 |
| old association (isotropic, plain Mahalanobis) + team colour | 59% | 32 | 3.9 | 12 | 22 | 20 | 4 |
