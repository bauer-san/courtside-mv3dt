| variant | runs | identity IDF1 on detected labels | labelled people detected | end-to-end: correctly identified / all labelled |
|---|---|---|---|---|
| INT8, detector threshold 0.2 (nvinfer default) | 3 | 78.9% (61%-92%) | 94.2% | **74.6%** (56%-89%) |
| INT8, detector threshold 0.1 | 3 | 79.6% (70%-87%) | 97.2% | **77.5%** (67%-86%) |
