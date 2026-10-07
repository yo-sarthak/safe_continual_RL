# Result tables

Tab-separated tables collected from the training and evaluation logs. One row per run or evaluation; costs are mean episodic cost, and `cost_over_d` is cost divided by the bound.

| File | Rows | Contents |
|---|---|---|
| `runs.tsv` | 942 | Every training run: configuration (level, bound, seed, arm, EWC strength `coef`), final reward and cost, and for phase-2 runs the retention (`retain_*`, old setting) and adaptation (`adapt_*`, new setting) results. |
| `sweeps.tsv` | 2,809 | Each trained policy evaluated across a range of bounds `threshold_d`. |
| `retention_curve.tsv` | 2,612 | Cost and reward on every setting seen so far, at four checkpoints per phase (`frac_of_phase`), for the two- and three-phase sequences. |
| `fisher.tsv` | 517 | Per-anchor importance-map statistics: share of zero-cost samples, cost-critic fit (`r2_*`), and how concentrated the Fisher mass is (`top1_pct`, `top10_pct`). |

The arm names in the `variant` column: `naive` is plain retraining, `vanilla` the standard Fisher, `realvb` the cost-aware Fisher, and `l2` uniform anchoring.
