### YOLOX vs PeopleNet boxes on js_r1 (time-aligned clips; YOLOX 640 x 640 letterboxed)

| detector, score threshold | player | referee | not a player | unclear | court-filtered boxes per frame | boxes on no PeopleNet box, per frame |
|---|---|---|---|---|---|---|
| PeopleNet (as run) | 100 % | 100 % | 100 % | 100 % | 5.73 | - |
| yolox_m, 0.1 | 97.0% (732) | 96.3% (109) | 92.3% (117) | 95.0% (60) | 8.85 | 3.23 |
| yolox_m, 0.25 | 95.9% (732) | 95.4% (109) | 71.8% (117) | 95.0% (60) | 7.57 | 2.18 |
| yolox_m, 0.4 | 95.1% (732) | 94.5% (109) | 16.2% (117) | 86.7% (60) | 6.59 | 1.60 |
| yolox_l, 0.1 | 97.1% (732) | 100.0% (109) | 96.6% (117) | 95.0% (60) | 8.78 | 3.07 |
| yolox_l, 0.25 | 96.3% (732) | 99.1% (109) | 60.7% (117) | 91.7% (60) | 7.78 | 2.36 |
| yolox_l, 0.4 | 95.4% (732) | 97.2% (109) | 14.5% (117) | 86.7% (60) | 6.94 | 1.89 |

**yolox_m, threshold 0.25: box placement (YOLOX minus PeopleNet, px, median) and cross-camera distance of box-bottom floor points (median, m)**

| camera | top | bottom | matched |
|---|---|---|---|
| NL | -0.5 | -3.3 | 923 |
| FL | -1.3 | -1.4 | 1790 |
| NR | -0.5 | -3.9 | 1314 |
| FR | -0.8 | -2.5 | 2032 |

| camera pair | pairs | PeopleNet | YOLOX |
|---|---|---|---|
| FL-FR | 93 | 0.35 | 0.28 |
| FL-NL | 720 | 0.76 | 0.63 |
| FL-NR | 11 | 0.29 | 0.27 |
| FR-NR | 1039 | 0.73 | 0.57 |
| all | 1863 | 0.73 | 0.58 |

**yolox_l, threshold 0.25: box placement (YOLOX minus PeopleNet, px, median) and cross-camera distance of box-bottom floor points (median, m)**

| camera | top | bottom | matched |
|---|---|---|---|
| NL | -0.9 | -2.8 | 922 |
| FL | -1.2 | -1.2 | 1806 |
| NR | -0.8 | -3.4 | 1320 |
| FR | -1.0 | -2.3 | 2030 |

| camera pair | pairs | PeopleNet | YOLOX |
|---|---|---|---|
| FL-FR | 93 | 0.37 | 0.29 |
| FL-NL | 730 | 0.76 | 0.66 |
| FL-NR | 10 | 0.28 | 0.26 |
| FR-NR | 1041 | 0.74 | 0.60 |
| all | 1874 | 0.73 | 0.61 |
