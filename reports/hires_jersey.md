| reads from | runs | torsos with text (game) | reads (game) | reads near labels: correct / wrong | court tracker IDF1 | end-to-end | switches per minute |
|---|---|---|---|---|---|---|---|
| jersey.npz | 3 | 12875 | 5409 | 2185 / 131 | 79.9% (70%-89%) | 77.7% | 28 |
| jersey_1080p_h150_nearest.npz | 3 | 12921 | 5507 | 2204 / 131 | 80.4% (68%-90%) | 78.2% | 25 |
| jersey_1080p_h150_area.npz | 3 | 13962 | 6556 | 2554 / 190 | 84.0% (79%-91%) | 81.7% | 26 |
| jersey_4k_h150_area.npz | 3 | 13294 | 6143 | 2404 / 171 | 79.2% (67%-92%) | 77.1% | 27 |

Per run and minute (court tracker IDF1): jersey.npz s_int8t10_r1 83% 81%; jersey.npz s_int8t10_r2 86% 70%; jersey.npz s_int8t10_r3 89% 71%; jersey_1080p_h150_nearest.npz s_int8t10_r1 90% 79%; jersey_1080p_h150_nearest.npz s_int8t10_r2 85% 72%; jersey_1080p_h150_nearest.npz s_int8t10_r3 89% 68%; jersey_1080p_h150_area.npz s_int8t10_r1 88% 81%; jersey_1080p_h150_area.npz s_int8t10_r2 86% 79%; jersey_1080p_h150_area.npz s_int8t10_r3 91% 79%; jersey_4k_h150_area.npz s_int8t10_r1 80% 72%; jersey_4k_h150_area.npz s_int8t10_r2 89% 76%; jersey_4k_h150_area.npz s_int8t10_r3 92% 67%
