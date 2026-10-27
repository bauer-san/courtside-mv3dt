| variant | runs | identity IDF1 on detected labels | labelled people detected | end-to-end: correctly identified / all labelled |
|---|---|---|---|---|
| FP16, every frame (with display) | 3 | 80.6% (71%-90%) | 98.8% | **79.7%** (70%-89%) |
| FP16, every frame, no display | 1 | 79.3% (72%-87%) | 99.0% | **78.5%** (71%-86%) |
| INT8, every frame | 3 | 76.3% (68%-85%) | 94.7% | **72.3%** (63%-83%) |
| INT8, every frame, detector threshold 0.1 | 3 | 76.5% (65%-84%) | 97.9% | **74.9%** (63%-83%) |
| FP16, every 2nd frame | 3 | 78.7% (75%-82%) | 82.9% | **65.2%** (60%-69%) |
| FP16, every 3rd frame | 3 | 76.8% (63%-89%) | 66.6% | **51.6%** (40%-63%) |
| INT8, every 2nd frame | 3 | 79.3% (77%-82%) | 75.4% | **59.7%** (57%-61%) |
