### Python reference, deployable (mount runs): mount_r1, mount_r2, mount_r3

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

### Live in DeepStream (ct2 runs): ct2_r1, ct2_r2, ct2_r3

| labelled as | detections | in the output | with an identity |
|---|---|---|---|
| player | 2228 | 2222 (100%) | 2084 (94%) |
| referee | 345 | 336 (97%) | 227 (66%) |
| not a player | 457 | 280 (61%) | 111 (24%) |
| unclear | 199 | 196 (98%) | 175 (88%) |

Not-a-player detections (457):
- position: 0% outside the lines, 17% within 1.5 m inside, 82% further in (median +1.99 m from the nearest line)
- in the output: 280; their tracks' 90th-percentile spread around the median position: median 8.14 m, 13% under 1 m
- identities given to them: A:1 x26, B:0 x24, A:3 x14, B:3 x13, A:22 x10, B:1 x7, A:94 x5, B:2 x5
- body height from the box (FL/FR crops): non-players median 1.48 m (n=442, 22% under 1.4 m); players median 1.81 m (n=1106, 3% under 1.4 m)
