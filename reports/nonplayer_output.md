### Python reference, default configuration: s_int8t10_r1, s_int8t10_r2, s_int8t10_r3

Identity IDF1 (players): 79.9% (70%-89%); per run-minute 83.1%, 80.8%, 85.8%, 69.8%, 89.2%, 70.7%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2171 | 2163 (100%) | 2072 (95%) |
| referee | 327 | 322 (98%) | 152 (46%) |
| not a player | 336 | 19 (6%) | 10 (3%) |
| unclear | 177 | 164 (93%) | 147 (83%) |

Not-a-player detections (336):
- position: 0% outside the lines, 32% within 1.5 m inside, 68% further in (median +1.93 m from the nearest line)
- in the output: 19; their tracks' 90th-percentile spread around the median position: median 8.24 m, 16% under 1 m
- identities given to them: A:3 x4, B:1 x4, A:94 x1, B:22 x1
- body height from the box (FL/FR crops): non-players median 1.50 m (n=300, 12% under 1.4 m); players median 1.83 m (n=540, 1% under 1.4 m)

### Live in DeepStream: cts_r1, cts_r2, cts_r3

Identity IDF1 (players): 82.2% (74%-89%); per run-minute 88.8%, 73.9%, 89.2%, 77.7%, 85.9%, 77.9%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2180 | 2171 (100%) | 2081 (95%) |
| referee | 329 | 323 (98%) | 136 (41%) |
| not a player | 336 | 21 (6%) | 6 (2%) |
| unclear | 179 | 167 (93%) | 155 (87%) |

Not-a-player detections (336):
- position: 1% outside the lines, 21% within 1.5 m inside, 78% further in (median +2.10 m from the nearest line)
- in the output: 21; their tracks' 90th-percentile spread around the median position: median 6.92 m, 14% under 1 m
- identities given to them: A:1 x4, A:94 x1, A:0 x1
- body height from the box (FL/FR crops): non-players median 1.50 m (n=300, 12% under 1.4 m); players median 1.83 m (n=539, 1% under 1.4 m)
