### Detection-level comparison with nodisp_r1 (FP16, detector every frame), game frames

| run | config | fps/camera | GPU use | GPU mem | frames with output | box recall vs ref | box precision vs ref | label recall |
|---|---|---|---|---|---|---|---|---|
| mount_r1 | extra= | 56.0 | 87 % | 2.69 GB | 100% | 98.6% | 98.6% | 98.7% |
| int1_r1 | display=0 pgie_interval=1 pgie_precision=fp16 extra=TargetManagement.earlyTerminationAge=3 | 74.4 | 76 % | 2.40 GB | 50% | 80.1% | 98.7% | 83.5% |
| int1_r2 | display=0 pgie_interval=1 pgie_precision=fp16 extra=TargetManagement.earlyTerminationAge=3 | 75.4 | 76 % | 2.50 GB | 50% | 80.0% | 98.7% | 82.9% |
| int1_r3 | display=0 pgie_interval=1 pgie_precision=fp16 extra=TargetManagement.earlyTerminationAge=3 | 75.6 | 77 % | 2.50 GB | 50% | 79.7% | 98.7% | 82.3% |
| int2_r1 | display=0 pgie_interval=2 pgie_precision=fp16 extra=TargetManagement.earlyTerminationAge=4 | 45.6 | 68 % | 2.42 GB | 33% | 65.1% | 99.2% | 67.1% |
| int2_r2 | display=0 pgie_interval=2 pgie_precision=fp16 extra=TargetManagement.earlyTerminationAge=4 | 45.1 | 68 % | 2.41 GB | 33% | 65.1% | 99.2% | 66.5% |
| int2_r3 | display=0 pgie_interval=2 pgie_precision=fp16 extra=TargetManagement.earlyTerminationAge=4 | 46.5 | 68 % | 2.33 GB | 33% | 65.1% | 99.2% | 65.9% |
| int8_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 94.4 | 88 % | 2.46 GB | 100% | 91.5% | 89.1% | 94.8% |
| int8_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 94.0 | 88 % | 2.46 GB | 100% | 91.5% | 88.7% | 94.7% |
| int8_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 95.4 | 89 % | 2.46 GB | 100% | 91.7% | 88.8% | 94.4% |
| int8int1_r1 | display=0 pgie_interval=1 pgie_precision=int8 extra=TargetManagement.earlyTerminationAge=3 | 92.4 | 77 % | 2.44 GB | 50% | 68.4% | 92.2% | 76.5% |
| int8int1_r2 | display=0 pgie_interval=1 pgie_precision=int8 extra=TargetManagement.earlyTerminationAge=3 | 92.6 | 76 % | 2.44 GB | 50% | 68.2% | 92.1% | 75.0% |
| int8int1_r3 | display=0 pgie_interval=1 pgie_precision=int8 extra=TargetManagement.earlyTerminationAge=3 | 92.2 | 76 % | 2.54 GB | 50% | 67.9% | 92.1% | 74.4% |
| int8t10_r1 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 93.8 | 90 % | 2.38 GB | 100% | 93.9% | 88.4% | 97.6% |
| int8t10_r2 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 65.0 | 83 % | 2.47 GB | 100% | 93.5% | 88.4% | 97.6% |
| int8t10_r3 | display=0 pgie_interval=0 pgie_precision=int8 extra= | 92.6 | 89 % | 2.47 GB | 100% | 93.6% | 88.5% | 98.2% |
