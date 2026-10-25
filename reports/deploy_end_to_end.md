| variant | runs | identity IDF1 on detected labels | labelled people detected | end-to-end: correctly identified / all labelled |
|---|---|---|---|---|
| FP16, every frame (with display) | 3 | 77.8% (72%-84%) | 98.8% | **76.9%** (71%-83%) |
| FP16, every frame, no display | 1 | 79.5% (72%-87%) | 99.0% | **78.7%** (72%-86%) |
| INT8, every frame | 3 | 77.1% (69%-82%) | 94.7% | **73.1%** (64%-79%) |
| INT8, every frame, detector threshold 0.1 | 3 | 75.2% (67%-83%) | 97.9% | **73.6%** (65%-82%) |
| FP16, every 2nd frame | 3 | 70.9% (56%-80%) | 82.9% | **58.7%** (47%-65%) |
| FP16, every 3rd frame | 3 | 75.9% (65%-86%) | 66.6% | **50.9%** (40%-60%) |
| INT8, every 2nd frame | 3 | 69.8% (66%-72%) | 75.4% | **52.6%** (50%-55%) |
