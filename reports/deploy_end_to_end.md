| variant | runs | identity IDF1 on detected labels | labelled people detected | end-to-end: correctly identified / all labelled |
|---|---|---|---|---|
| FP16, every frame (with display) | 3 | 79.2% (74%-89%) | 98.8% | **78.3%** (73%-88%) |
| FP16, every frame, no display | 1 | 75.6% (70%-81%) | 99.0% | **74.8%** (69%-80%) |
| INT8, every frame | 3 | 75.2% (69%-79%) | 94.7% | **71.2%** (64%-76%) |
| INT8, every frame, detector threshold 0.1 | 3 | 74.2% (61%-83%) | 97.9% | **72.7%** (59%-82%) |
| FP16, every 2nd frame | 3 | 73.3% (67%-79%) | 82.9% | **60.7%** (56%-67%) |
| FP16, every 3rd frame | 3 | 73.9% (65%-82%) | 66.6% | **49.5%** (41%-58%) |
| INT8, every 2nd frame | 3 | 73.6% (70%-77%) | 75.4% | **55.5%** (52%-59%) |
