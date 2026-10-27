### Python reference, main configuration (mount runs) (without the seated-person rule): mount_r1, mount_r2, mount_r3

Identity IDF1 (players): 79.8% (74%-88%); per run-minute 83.4%, 76.0%, 88.3%, 77.7%, 79.1%, 74.4%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2227 | 2221 (100%) | 2099 (94%) |
| referee | 343 | 334 (97%) | 130 (38%) |
| not a player | 453 | 278 (61%) | 93 (21%) |
| unclear | 200 | 197 (98%) | 173 (86%) |

Not-a-player detections (453):
- position: 0% outside the lines, 23% within 1.5 m inside, 77% further in (median +1.99 m from the nearest line)
- in the output: 278; their tracks' 90th-percentile spread around the median position: median 2.83 m, 23% under 1 m
- identities given to them: B:0 x39, A:1 x26, B:3 x8, A:94 x8, A:3 x6, B:1 x3, A:7 x3
- body height from the box (FL/FR crops): non-players median 1.48 m (n=438, 23% under 1.4 m); players median 1.81 m (n=1104, 3% under 1.4 m)
