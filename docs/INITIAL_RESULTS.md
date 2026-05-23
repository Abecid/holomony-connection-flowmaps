# Initial quantitative results

These are **quick CPU-scale runs**, not the final paper numbers. They are included so you can sanity-check the repo immediately and have a first table to discuss.

Reproduce them with:

```bash
DEVICE=auto bash scripts/reproduce_shipped_initial.sh
```

## Noncommuting nonlinear connection

Setup: `world=nonlinear`, 120 training steps, batch 64, hidden 48, depth 2, seed 0. This is deliberately tiny; the goal is not SOTA, it is to check whether the holonomy loss creates the expected signal.

| model | local_mse | AB/BA comm MSE ↓ | AB/BA comm cosine ↑ | loop holonomy MSE ↓ | loop holonomy cosine ↑ | OOD comm MSE ↓ | OOD holonomy MSE ↓ | OOD holonomy cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat_pifm` | 0.02872 | 0.01531 | 0.3897 | 0.02978 | 0.3658 | 0.2567 | 0.2890 | 0.3237 |
| `holonomy_connection` | 0.01656 | **0.001886** | **0.9564** | **0.003869** | **0.9544** | **0.04238** | **0.07315** | **0.9134** |
| `independent_cfm` | 0.03532 | 0.02081 | 0.5968 | 0.03129 | 0.5961 | 0.3038 | 0.2190 | 0.7382 |
| `local_connection` | **0.01346** | 0.006069 | 0.8737 | 0.01409 | 0.8884 | 0.1012 | 0.1236 | 0.8883 |
| `shared_cfm` | 0.01824 | 0.01107 | 0.6261 | 0.02442 | 0.6788 | 0.1882 | 0.2499 | 0.5829 |

Interpretation: `local_connection` fits local velocities slightly better, but the full `holonomy_connection` gives much better finite-time commutator and loop-holonomy alignment. `flat_pifm` suppresses the nonzero holonomy, which is exactly the intended foil.

## Commuting sanity check

Setup: `world=commute`, 80 training steps, same small model. True loop holonomy is zero. A credible model should not hallucinate large residuals.

| model | local_mse | AB/BA comm MSE ↓ | loop holonomy MSE ↓ | predicted loop norm ↓ | OOD loop MSE ↓ | OOD predicted loop norm ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `flat_pifm` | 1.748e-4 | **4.409e-7** | **5.981e-7** | **8.853e-4** | **8.368e-6** | **3.572e-3** |
| `holonomy_connection` | **1.732e-4** | 4.316e-6 | 5.342e-6 | 2.489e-3 | 6.669e-5 | 1.092e-2 |
| `local_connection` | 1.967e-4 | 1.011e-5 | 1.066e-5 | 3.253e-3 | 2.435e-4 | 1.822e-2 |
| `shared_cfm` | 1.861e-4 | 1.257e-5 | 1.977e-5 | 4.496e-3 | 3.334e-4 | 2.146e-2 |

Interpretation: the holonomy model does **not** produce large spurious curvature in a flat world. `flat_pifm` is best here, as expected, because its zero-curvature bias is correct in the commuting setting.

## Scaling diagnostic

For the noncommuting run, `holonomy_connection` tracks the true loop residual over loop sizes, while `flat_pifm` underpredicts it.

`results/initial/holonomy_scaling.csv` and `results/initial/flat_pifm_scaling.csv` contain the raw numbers.

## Caveat

These are first-pass synthetic results. They are good enough to share as an initial sanity table, but before writing a workshop submission you should run:

1. 3 random seeds.
2. Longer runs: `STEPS=1000+`, hidden 96/128.
3. The nonlinear stress test and the commuting sanity check.
4. A small image-space follow-up: affine-MNIST or dSprites with ordered transforms.
