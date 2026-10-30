### Detection-level comparison with s_int8_r1, game frames

| run | config | fps/camera | GPU use | GPU mem | frames with output | box recall vs ref | box precision vs ref | label recall |
|---|---|---|---|---|---|---|---|---|
| s_int8_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 96.0 | 89 % | 2.38 GB | 100% | 97.6% | 98.1% | 76.1% |
| s_int8_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 96.2 | 88 % | 2.28 GB | 100% | 98.3% | 98.4% | 75.8% |
| s_int8t10_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 95.7 | 88 % | 2.28 GB | 100% | 98.3% | 94.8% | 78.1% |
| s_int8t10_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 93.8 | 89 % | 2.40 GB | 100% | 97.6% | 95.3% | 77.8% |
| s_int8t10_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 96.9 | 89 % | 2.42 GB | 100% | 98.0% | 95.2% | 78.0% |
| np_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 161.8 | 99 % | 1.51 GB | 100% | 98.0% | 95.0% | 77.4% |
| np_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 159.8 | 99 % | 1.40 GB | 100% | 98.1% | 94.9% | 77.7% |
| np_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 156.6 | 97 % | 1.43 GB | 100% | 97.9% | 95.1% | 77.7% |
| npct_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 146.4 | 96 % | 1.62 GB | 100% | 98.0% | 94.9% | 77.5% |
| npct_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 154.4 | 99 % | 1.53 GB | 100% | 98.1% | 94.9% | 77.7% |
| npct_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 144.9 | 98 % | 1.65 GB | 100% | 98.0% | 95.2% | 77.7% |
