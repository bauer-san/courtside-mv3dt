| Configuration | Identity IDF1 | Paired difference | Pairs better / worse |
|---|---|---|---|
| **main (causal): fixed bias from installation, online 3-cluster team model, no spectator filter ¹** | **80.6%** (71%-90%) | — | — |
| camera bias: estimated over the whole run (offline) | 79.8% (71%-90%) | -0.8 | 0 / 4 |
| camera bias: estimated live from the first 2 minutes | 80.5% (71%-91%) | -0.1 | 2 / 2 |
| camera bias: none | 78.3% (69%-88%) | -2.3 | 0 / 6 |
| team colours: online model, 4 clusters (two teams, referees, other) | 72.6% (69%-80%) | -8.0 | 0 / 6 |
| team colours: offline model, clustered over the game, clusters named by hand | 81.0% (74%-90%) | +0.4 | 2 / 2 |
| team colours: offline model from the first 2 minutes only ² | 79.3% (71%-89%) | -1.3 | 0 / 2 |
| spectator filter and post-stitching (offline, whole tracks) | 80.5% (71%-90%) | -0.1 | 0 / 1 |
| all offline: whole-run bias, hand-named team model, spectator filter (upper reference) | 79.2% (74%-89%) | -1.4 | 2 / 3 |

¹ Seed 0 of the online model's first clustering (all rows); seeds 0-4: 80.6%, 80.6%, 80.6%, 80.6%, 80.6% (mean 80.6%).
² Clusters named after the nearest hand-named cluster, standing in for an operator.
Runs: mount_r1, mount_r2, mount_r3; fixed camera bias measured on ct2_r1, ct2_r2, ct2_r3 (not evaluated here).
