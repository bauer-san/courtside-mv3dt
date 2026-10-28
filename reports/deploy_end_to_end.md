| variant | runs | identity IDF1 on detected labels | labelled people detected | end-to-end: correctly identified / all labelled |
|---|---|---|---|---|
| INT8, detector threshold 0.2 (nvinfer default) | 3 | 78.5% (62%-91%) | 94.2% | **74.2%** (57%-88%) |
| INT8, detector threshold 0.1 | 3 | 79.9% (70%-89%) | 97.2% | **77.7%** (67%-88%) |
