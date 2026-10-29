### Python reference, default configuration: js_r1, js_r2, js_r3

Identity IDF1 (players): 85.6% (73%-96%); per run-minute 95.0%, 72.7%, 93.1%, 83.7%, 96.3%, 73.0%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2180 | 2170 (100%) | 2075 (95%) |
| referee | 328 | 322 (98%) | 175 (53%) |
| not a player | 338 | 19 (6%) | 0 (0%) |
| unclear | 179 | 165 (92%) | 159 (89%) |

Not-a-player detections (338):
- position: 1% outside the lines, 21% within 1.5 m inside, 79% further in (median +1.95 m from the nearest line)
- in the output: 19; their tracks' 90th-percentile spread around the median position: median 10.55 m, 26% under 1 m
- body height from the box (FL/FR crops): non-players median 1.51 m (n=301, 12% under 1.4 m); players median 1.84 m (n=537, 1% under 1.4 m)

### Live in DeepStream: ctj_r1, ctj_r2, ctj_r3

Identity IDF1 (players): 84.4% (72%-96%); per run-minute 95.6%, 74.8%, 96.3%, 73.5%, 94.4%, 71.6%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2181 | 2172 (100%) | 2106 (97%) |
| referee | 326 | 320 (98%) | 153 (47%) |
| not a player | 332 | 26 (8%) | 7 (2%) |
| unclear | 178 | 169 (95%) | 154 (87%) |

Not-a-player detections (332):
- position: 1% outside the lines, 34% within 1.5 m inside, 65% further in (median +1.91 m from the nearest line)
- in the output: 26; their tracks' 90th-percentile spread around the median position: median 2.03 m, 31% under 1 m
- identities given to them: B:0 x4, B:2 x1, A:2 x1, A:3 x1
- body height from the box (FL/FR crops): non-players median 1.50 m (n=295, 12% under 1.4 m); players median 1.84 m (n=536, 1% under 1.4 m)
