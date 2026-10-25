### Python reference, deployable (mount runs) (without the seated-person rule): mount_r1, mount_r2, mount_r3

Identity IDF1 (players): 78.9% (69%-88%); per run-minute 87.7%, 71.3%, 87.7%, 77.7%, 79.7%, 69.5%

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2227 | 2221 (100%) | 2086 (94%) |
| referee | 343 | 334 (97%) | 161 (47%) |
| not a player | 453 | 278 (61%) | 66 (15%) |
| unclear | 200 | 197 (98%) | 177 (88%) |

Not-a-player detections (453):
- position: 0% outside the lines, 23% within 1.5 m inside, 77% further in (median +1.99 m from the nearest line)
- in the output: 278; their tracks' 90th-percentile spread around the median position: median 3.15 m, 21% under 1 m
- identities given to them: A:1 x15, B:0 x15, A:3 x13, A:22 x10, A:94 x5, B:3 x4, A:7 x3, B:1 x1
- body height from the box (FL/FR crops): non-players median 1.48 m (n=438, 23% under 1.4 m); players median 1.81 m (n=1104, 3% under 1.4 m)
