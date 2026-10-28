### Python reference, default configuration: js_r1, js_r2, js_r3

Identity IDF1 (players): 85.3% (74%-96%); per run-minute 95.0%, 74.2%, 93.1%, 79.5%, 96.3%, 73.6%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2180 | 2170 (100%) | 2063 (95%) |
| referee | 328 | 322 (98%) | 175 (53%) |
| not a player | 338 | 19 (6%) | 0 (0%) |
| unclear | 179 | 165 (92%) | 159 (89%) |

Not-a-player detections (338):
- position: 1% outside the lines, 21% within 1.5 m inside, 79% further in (median +1.95 m from the nearest line)
- in the output: 19; their tracks' 90th-percentile spread around the median position: median 10.55 m, 26% under 1 m
- body height from the box (FL/FR crops): non-players median 1.51 m (n=301, 12% under 1.4 m); players median 1.84 m (n=537, 1% under 1.4 m)

### Live in DeepStream: ctj_r1, ctj_r2, ctj_r3

Identity IDF1 (players): 83.1% (67%-96%); per run-minute 93.5%, 73.3%, 96.0%, 73.2%, 95.0%, 67.5%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2181 | 2172 (100%) | 2093 (96%) |
| referee | 326 | 320 (98%) | 153 (47%) |
| not a player | 332 | 26 (8%) | 7 (2%) |
| unclear | 178 | 169 (95%) | 154 (87%) |

Not-a-player detections (332):
- position: 1% outside the lines, 34% within 1.5 m inside, 65% further in (median +1.91 m from the nearest line)
- in the output: 26; their tracks' 90th-percentile spread around the median position: median 2.03 m, 31% under 1 m
- identities given to them: B:0 x4, B:2 x1, A:2 x1, B:3 x1
- body height from the box (FL/FR crops): non-players median 1.50 m (n=295, 12% under 1.4 m); players median 1.84 m (n=536, 1% under 1.4 m)
