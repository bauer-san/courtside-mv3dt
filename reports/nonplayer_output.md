### Python reference, main configuration (mount runs): mount_r1, mount_r2, mount_r3

Identity IDF1 (players): 80.6% (71%-90%); per run-minute 87.1%, 75.1%, 90.2%, 75.4%, 84.6%, 71.2%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2227 | 2212 (99%) | 2114 (95%) |
| referee | 343 | 329 (96%) | 148 (43%) |
| not a player | 453 | 17 (4%) | 0 (0%) |
| unclear | 200 | 184 (92%) | 169 (84%) |

Not-a-player detections (453):
- position: 0% outside the lines, 23% within 1.5 m inside, 77% further in (median +1.99 m from the nearest line)
- in the output: 17; their tracks' 90th-percentile spread around the median position: median 11.20 m, 0% under 1 m
- body height from the box (FL/FR crops): non-players median 1.48 m (n=438, 23% under 1.4 m); players median 1.81 m (n=1104, 3% under 1.4 m)

### Live in DeepStream: ct3_r1, ct3_r2, ct3_r3

Identity IDF1 (players): 80.0% (71%-87%); per run-minute 87.1%, 74.1%, 84.4%, 71.4%, 86.9%, 76.2%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2220 | 2205 (99%) | 2082 (94%) |
| referee | 344 | 325 (94%) | 185 (54%) |
| not a player | 455 | 16 (4%) | 1 (0%) |
| unclear | 198 | 178 (90%) | 163 (82%) |

Not-a-player detections (455):
- position: 0% outside the lines, 15% within 1.5 m inside, 85% further in (median +2.00 m from the nearest line)
- in the output: 16; their tracks' 90th-percentile spread around the median position: median 10.74 m, 0% under 1 m
- identities given to them: B:3 x1
- body height from the box (FL/FR crops): non-players median 1.48 m (n=440, 22% under 1.4 m); players median 1.81 m (n=1099, 3% under 1.4 m)
