# Experimental Design

## Purpose

Show that holonomy supervision improves finite-time order-sensitive rollouts in controlled generative dynamics.

## Main claim

Local vector-field matching is not always enough. A model can have reasonable local velocity error but poor finite-time loop behavior. Explicitly matching finite-time holonomy improves loop residuals and OOD ordered compositions.

## Why the nonlinear world exists

Plain rotation + translation is too easy: independent `V_A(x)`, `V_B(x)` can compose to noncommuting endpoints. The nonlinear world makes both directions depend on control coordinates and state, producing a genuine connection over control space.

## Baselines

1. `independent_cfm`: tests whether independent primitive controls are enough.
2. `shared_cfm`: tests whether local controlled velocity regression is enough.
3. `local_connection`: tests whether rollout supervision without explicit loop residual matching is enough.
4. `flat_pifm`: tests the opposite inductive bias: zero curvature/path independence.
5. `holonomy_connection`: full method.

## Kill conditions

Abandon or reformulate if:

- `holonomy_connection` does not beat `shared_cfm` on `id_hol_mse`.
- `holonomy_connection` hallucinates large `id_hol_pred_norm` in `world=commute`.
- `local_connection` always matches `holonomy_connection`; then holonomy loss is unnecessary.
- Results only work on `se2` but not `nonlinear`.

## Next non-synthetic experiment

After the synthetic result is clean, implement affine-MNIST or dSprites ordered image transforms:

- rotate then translate vs translate then rotate,
- rotate then shear vs shear then rotate,
- loop residual images.

The same training/eval API can be reused by replacing `worlds.py` and the rollout target generator.
