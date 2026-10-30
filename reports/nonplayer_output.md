### Python reference, default configuration: np_r1, np_r2, np_r3

Identity IDF1 (players): 84.4% (72%-96%); per run-minute 95.6%, 73.9%, 96.0%, 72.5%, 93.8%, 74.6%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2170 | 2155 (99%) | 2051 (95%) |
| referee | 327 | 321 (98%) | 194 (59%) |
| not a player | 331 | 12 (4%) | 0 (0%) |
| unclear | 179 | 164 (92%) | 157 (88%) |

Not-a-player detections (331):
- position: 0% outside the lines, 5% within 1.5 m inside, 95% further in (median +2.09 m from the nearest line)
- in the output: 12; their tracks' 90th-percentile spread around the median position: median 1.56 m, 0% under 1 m
- body height from the box (FL/FR crops): non-players median 1.50 m (n=294, 12% under 1.4 m); players median 1.84 m (n=530, 1% under 1.4 m)

### Live in DeepStream: up_r1, up_r2, up_r3

Identity IDF1 (players): 88.6% (79%-96%); per run-minute 95.3%, 85.2%, 95.7%, 81.7%, 94.7%, 79.0%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2172 | 2157 (99%) | 2106 (97%) |
| referee | 327 | 321 (98%) | 146 (45%) |
| not a player | 330 | 12 (4%) | 0 (0%) |
| unclear | 179 | 164 (92%) | 154 (86%) |

Not-a-player detections (330):
- position: 0% outside the lines, 5% within 1.5 m inside, 95% further in (median +2.09 m from the nearest line)
- in the output: 12; their tracks' 90th-percentile spread around the median position: median 1.56 m, 0% under 1 m
- body height from the box (FL/FR crops): non-players median 1.50 m (n=294, 12% under 1.4 m); players median 1.84 m (n=531, 1% under 1.4 m)
