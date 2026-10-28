| videos | runs | MV3DT IDF1 | MV3DT switches | court tracker IDF1 | court tracker switches | court tracker end-to-end | NL-FL distance | FL-FR distance | NR-FR distance | FL-NR distance | NL-FR distance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original clips (no offsets) | int8t10_r1, int8t10_r2, int8t10_r3 | 54.8% (48%-62%) | 34 | 77.0% (68%-85%) | 33 | 75.4% | 0.60 m | 0.41 m | 0.40 m | 0.30 m | 0.56 m |
| clip start offsets (NL 26, FL 13, NR 4) | s_int8t10_r1, s_int8t10_r2, s_int8t10_r3 | 61.5% (59%-66%) | 29 | 79.9% (70%-89%) | 28 | 77.7% | 0.51 m | 0.36 m | 0.44 m | 0.23 m | 0.21 m |
| clip start offsets (FL 13, NR 22, FR 26) | tol_2x_r1 | 52.5% (50%-55%) | 36 | 72.8% (62%-84%) | 32 | 70.8% | 0.69 m | 0.58 m | 0.39 m | 0.41 m | 0.63 m |

Per run and minute (court tracker IDF1 / MV3DT IDF1): int8t10_r1 2to3 83% / 49%; int8t10_r1 4to5 75% / 61%; int8t10_r2 2to3 85% / 48%; int8t10_r2 4to5 71% / 55%; int8t10_r3 2to3 81% / 53%; int8t10_r3 4to5 68% / 62%; s_int8t10_r1 2to3 83% / 66%; s_int8t10_r1 4to5 81% / 61%; s_int8t10_r2 2to3 86% / 59%; s_int8t10_r2 4to5 70% / 61%; s_int8t10_r3 2to3 89% / 63%; s_int8t10_r3 4to5 71% / 59%; tol_2x_r1 2to3 84% / 50%; tol_2x_r1 4to5 62% / 55%
