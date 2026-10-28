### Detection-level comparison with s_int8_r1, game frames

| run | config | fps/camera | GPU use | GPU mem | frames with output | box recall vs ref | box precision vs ref | label recall |
|---|---|---|---|---|---|---|---|---|
| s_int8_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 96.0 | 89 % | 2.38 GB | 100% | 97.6% | 98.1% | 76.1% |
| s_int8_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 96.2 | 88 % | 2.28 GB | 100% | 98.3% | 98.4% | 75.8% |
| s_int8t10_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 95.7 | 88 % | 2.28 GB | 100% | 98.3% | 94.8% | 78.1% |
| s_int8t10_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 93.8 | 89 % | 2.40 GB | 100% | 97.6% | 95.3% | 77.8% |
| s_int8t10_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 96.9 | 89 % | 2.42 GB | 100% | 98.0% | 95.2% | 78.0% |
| js_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 89.0 | 89 % | 2.35 GB | 100% | 98.3% | 95.2% | 78.1% |
| js_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 94.3 | 90 % | 2.36 GB | 100% | 97.8% | 95.0% | 77.7% |
| js_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 94.8 | 90 % | 2.34 GB | 100% | 97.9% | 94.8% | 77.8% |
| ctj_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 91.4 | 88 % | 2.43 GB | 100% | 97.9% | 95.0% | 77.8% |
| ctj_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 91.4 | 89 % | 2.36 GB | 100% | 98.1% | 95.2% | 77.7% |
| ctj_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 90.7 | 89 % | 2.46 GB | 100% | 98.0% | 95.2% | 77.5% |
