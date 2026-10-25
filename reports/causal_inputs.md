| Input replaced | Identity IDF1 | Paired difference | Pairs better / worse |
|---|---|---|---|
| default: whole-run bias, hand-named team model, spectator filter | 79.2% (74%-89%) | — | — |
| camera bias: fixed, measured on another run (as at installation) | 80.3% (76%-89%) | +1.0 | 2 / 1 |
| camera bias: estimated live from the first 2 minutes | 79.3% (72%-86%) | +0.1 | 2 / 4 |
| camera bias: none | 76.5% (68%-86%) | -2.8 | 1 / 5 |
| team colours: clustered from the first 2 minutes only ¹ | 79.2% (70%-88%) | -0.1 | 2 / 3 |
| team colours: automatic online model, 3 clusters (two teams, everyone else) | 79.7% (71%-90%) | +0.4 | 3 / 2 |
| team colours: automatic online model, 4 clusters (two teams, referees, other) | 69.9% (64%-73%) | -9.4 | 0 / 6 |
| spectator filter: none | 79.3% (74%-90%) | +0.1 | 1 / 0 |
| all three: fixed bias + online 3-cluster model + no spectator filter ² | 80.4% (72%-90%) | +1.2 | 3 / 2 |

¹ Clusters named after the nearest hand-named cluster, standing in for an operator.
² Mean over 5 seeds of the online model's first clustering (per-seed means 80.4%, 80.4%, 80.4%, 80.4%, 80.4%); paired differences averaged over the seeds. Other rows: seed 0.
Runs: mount_r1, mount_r2, mount_r3; fixed bias of each run measured on: mount_r1 <- mount_r2, mount_r2 <- mount_r3, mount_r3 <- mount_r1.
