| Input replaced | Identity IDF1 | Paired difference | Pairs better / worse |
|---|---|---|---|
| default: whole-run bias, hand-named team model, spectator filter | 77.8% (72%-84%) | — | — |
| camera bias: fixed, measured on another run (as at installation) | 77.5% (72%-83%) | -0.3 | 2 / 3 |
| camera bias: estimated live from the first 2 minutes | 76.7% (70%-84%) | -1.0 | 1 / 3 |
| camera bias: none | 77.3% (70%-81%) | -0.5 | 4 / 2 |
| team colours: clustered from the first 2 minutes only ¹ | 76.7% (71%-84%) | -1.1 | 2 / 4 |
| team colours: automatic online model, 3 clusters (two teams, everyone else) | 80.4% (69%-90%) | +2.6 | 4 / 1 |
| team colours: automatic online model, 4 clusters (two teams, referees, other) | 71.0% (54%-82%) | -6.7 | 1 / 4 |
| spectator filter: none | 77.8% (72%-84%) | +0.0 | 0 / 0 |
| all three: fixed bias + online 3-cluster model + no spectator filter ² | 79.0% (69%-90%) | +1.2 | 3 / 2 |

¹ Clusters named after the nearest hand-named cluster, standing in for an operator.
² Mean over 5 seeds of the online model's first clustering (per-seed means 78.9%, 78.9%, 78.9%, 78.9%, 79.2%); paired differences averaged over the seeds. Other rows: seed 0.
Runs: mount_r1, mount_r2, mount_r3; fixed bias of each run measured on: mount_r1 <- mount_r2, mount_r2 <- mount_r3, mount_r3 <- mount_r1.
