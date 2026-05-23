# Holonomy Connection Flow Maps

Controlled-connection flow maps for **order-sensitive / noncommuting generative dynamics**.

This repo implements the full method we actually want to test, not just endpoint regression with a curvature-sounding name. The learned object is a pair of controlled vector fields

\[
\frac{dx}{d\alpha}=V_A^\theta(x,\alpha,\beta),\qquad
\frac{dx}{d\beta}=V_B^\theta(x,\alpha,\beta),
\]

whose finite-time rollouts can have nonzero loop holonomy:

\[
H_{AB}(x)=F_B^{-\Delta}F_A^{-\Delta}F_B^\Delta F_A^\Delta(x)-x.
\]

The core hypothesis is:

> Path-independent multi-parameter flows are the wrong inductive bias when control order is meaningful. In such settings, nonzero finite-time holonomy is signal, not error.

This is the direct experimental foil to path-independent multi-parameter flow methods: flat models try to kill curvature; this method learns it.

---

## 1. Setup

### Conda CPU / debug setup

```bash
conda env create -f environment.yml
conda activate holonomy-connection-flowmaps
pip install -e .
```

### CUDA / A800 setup 

For the 2x A800 box, use the CUDA environment:

```bash
conda env create -f environment-cuda.yml
conda activate holonomy-connection-flowmaps
pip install -e .
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

The training code is PyTorch-native and supports CUDA, TF32, and AMP/BF16:

```bash
python -m holoflow_conn.train_connection --device cuda --amp --amp-dtype bf16 --tf32 ...
```

For the small synthetic MLP benchmark, the fastest use of two A800s is usually **parallel independent runs**, not `DataParallel`: each baseline/model runs on one GPU. For the image benchmark, batch sizes of 512--2048 are reasonable starting points.


### Pip-only alternative

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Mac note

On Apple Silicon, install PyTorch according to the official PyTorch selector, then run:

```bash
pip install -e .
```


---

## 1.1 W&B logging and held-out paper metrics

Both training entry points support optional Weights & Biases logging:

```bash
wandb login
```

Synthetic controlled-connection run:

```bash
python -m holoflow_conn.train_connection \
  --model holonomy_connection \
  --world nonlinear \
  --steps 3000 \
  --batch-size 256 \
  --eval-every 500 \
  --wandb \
  --wandb-project holonomy-connection-flowmaps \
  --wandb-group synthetic_nonlinear_seed0 \
  --wandb-run-name holonomy_connection_nonlinear_seed0
```

Affine-MNIST run:

```bash
python -m holoflow_conn.affine_mnist \
  --model holonomy_connection \
  --data-dir data \
  --steps 8000 \
  --batch-size 1024 \
  --eval-every 500 \
  --amp --amp-dtype bf16 \
  --wandb \
  --wandb-project holonomy-connection-flowmaps \
  --wandb-group affine_mnist_seed0 \
  --wandb-run-name affine_holonomy_connection_seed0
```

The training loop now runs the held-out evaluation suite every `--eval-every` steps. `--test-every` is accepted as an alias. Default is 500 steps.

The most paper-relevant metrics are logged under `paper/*`:

```text
paper/local_mse
paper/mse_random
paper/mse_ab_ba
paper/comm_mse
paper/comm_cosine
paper/comm_true_norm
paper/comm_pred_norm
paper/loop_holonomy_mse
paper/loop_holonomy_cosine
paper/loop_pred_identity_mse
paper/ood_comm_mse
paper/ood_comm_cosine
paper/ood_loop_holonomy_mse
paper/ood_loop_holonomy_cosine
paper/flatness_norm2
```

Raw metrics are also logged under `test/*`, training losses under `train/*`, and best-so-far model-selection values under `best/*`. By default, training writes `best_comm_mse.pt`, `best_comm_cosine.pt`, `best_loop_holonomy_mse.pt`, and `best_ood_loop_holonomy_mse.pt` when those criteria improve. For `world=commute`, it also tracks `best_loop_pred_identity_mse.pt`, because low identity residual means the model is not hallucinating curvature. In noncommuting worlds, identity residual is logged as a diagnostic curve but is **not** used for model selection, because selecting for low identity residual would reward suppressing true holonomy. Use `--no-save-best-checkpoints` to disable best-checkpoint writes.

For the two-GPU scripts, enable W&B with environment variables:

```bash
WANDB=1 WANDB_PROJECT=holonomy-connection-flowmaps TEST_EVERY=500 \
  STEPS=10000 BATCH=4096 HIDDEN=256 DEPTH=5 \
  bash scripts/run_a800_synthetic_two_gpu.sh
```

For offline logging on a cluster:

```bash
WANDB=1 WANDB_MODE=offline bash scripts/run_a800_synthetic_two_gpu.sh
```

Upload config/CSV/checkpoint artifacts by adding:

```bash
WANDB_ARTIFACTS=1
```

---

## 2. Dataset setup

No external dataset is needed for the first publishable synthetic experiment. The repo generates controlled trajectories on the fly.

The main benchmark is a known 2D **nonlinear non-Abelian connection**. States are sampled from an asymmetric Gaussian mixture. The ground-truth controlled fields are known:

```text
world=nonlinear
  A: nonlinear rotation/swirl field depending on beta
  B: control-dependent translation + shear + nonlinear bend depending on alpha/beta
```

This is deliberately stronger than plain rotation + translation. Independent CFM can solve plain rotation/translation by learning `F_A` and `F_B` separately, so that toy alone is not enough. The nonlinear world forces the model to learn a genuine control-dependent connection.

Also included:

```text
world=se2       # easy noncommuting sanity check: rotation + translation
world=body      # body-frame / alpha-dependent translation
world=commute   # negative control: x-translation + y-translation, true holonomy zero
```

---

## 3. Baselines included

All baselines are in this repo.

| Model | Meaning | Losses by default |
|---|---|---|
| `independent_cfm` | learns `V_A(x)` and `V_B(x)` separately; no control coordinates | local velocity loss |
| `shared_cfm` | learns `V(x, alpha, beta, direction)` | local velocity loss |
| `local_connection` | shared controlled field plus finite rollout supervision | local + rollout + small cycle |
| `flat_pifm` | shared controlled field with zero-curvature / path-independence regularizer | local + flatness + small cycle |
| `holonomy_connection` | full method | local + rollout + holonomy + small cycle |

The important comparison is **not** just against a dumb endpoint model. The real test is whether explicit holonomy supervision improves finite-time loop/order behavior over local CFM and flat/path-independent baselines.

---

## 4. Smoke test

```bash
bash scripts/run_smoke.sh
```

This trains for 20 steps and verifies the code path.

---

## 5. Main synthetic experiment

Fast CPU-ish version, one Python process, easiest to debug:

```bash
STEPS=150 BATCH=96 HIDDEN=64 DEPTH=3 DEVICE=auto bash scripts/run_quick_benchmark.sh
```

The shipped tiny-run results are summarized in `docs/INITIAL_RESULTS.md` and stored under `results/`.

More serious run:

```bash
STEPS=3000 BATCH=256 HIDDEN=128 DEPTH=4 DEVICE=auto bash scripts/run_synthetic_publishable.sh
```

Outputs:

```text
runs/quick_benchmark_nonlinear/comparison.csv
runs/synthetic_nonlinear/<model>/metrics.csv
runs/synthetic_nonlinear/<model>/eval_metrics.csv
runs/synthetic_nonlinear/<model>/holonomy_scaling.csv
runs/synthetic_nonlinear/comparison.csv
```

---

## 6. Commuting sanity experiment

This is the negative control. True holonomy should be zero.

```bash
STEPS=1500 BATCH=256 DEVICE=auto bash scripts/run_commute_sanity.sh
```

A credible method should **not hallucinate curvature** here. In particular:

```text
id_hol_pred_norm should be near zero
id_false_identity_mse should be near zero
```

---

## 7. Manual training commands

Train the full method:

```bash
python -m holoflow_conn.train_connection \
  --model holonomy_connection \
  --world nonlinear \
  --steps 3000 \
  --batch-size 256 \
  --hidden 128 \
  --depth 4 \
  --model-steps 6 \
  --gt-steps 16 \
  --eval-every 500 \
  --eval-batch-size 2048 \
  --eval-batches 2 \
  --outdir runs/manual/holonomy_connection \
  --device auto
```

Train a baseline:

```bash
python -m holoflow_conn.train_connection \
  --model shared_cfm \
  --world nonlinear \
  --steps 3000 \
  --batch-size 256 \
  --outdir runs/manual/shared_cfm
```

Evaluate a checkpoint:

```bash
python -m holoflow_conn.eval_connection \
  --checkpoint runs/manual/holonomy_connection/checkpoint.pt \
  --batch-size 4096 \
  --batches 4 \
  --gt-steps 32 \
  --scaling
```

Compare runs:

```bash
python -m holoflow_conn.compare_runs \
  --runs runs/manual/* \
  --out runs/manual/comparison.csv
```

---


---


## 8. Metrics to care about

### Local velocity fit

```text
local_mse
```

Checks whether the model learned the short-time fields. This alone is insufficient.

### Ordered composition

```text
ab_ba_endpoint_mse
ab_ba_comm_mse
ab_ba_comm_true_norm
ab_ba_comm_pred_norm
ab_ba_comm_cosine
```

These measure whether the model correctly predicts that `AB` and `BA` differ.

### Loop holonomy

```text
id_hol_mse
id_hol_true_norm
id_hol_pred_norm
id_hol_cosine
id_false_identity_mse
```

This is the core. The model should match the true loop residual, not collapse it to zero.

### OOD finite-time behavior

```text
ood_ab_ba_endpoint_mse
ood_ab_ba_comm_mse
ood_hol_mse
```

These evaluate larger held-out control magnitudes than training.

### Flatness diagnostic

```text
flatness_norm2
```

This estimates the squared curvature norm

\[
\|\partial_\alpha V_B-\partial_\beta V_A + [V_A,V_B]\|^2.
\]

`flat_pifm` should reduce this. On the nonlinear world, reducing this too much should hurt holonomy.

---

## 9. What counts as evidence that the method works?

The method is worth writing up if the nonlinear experiment shows:

1. `holonomy_connection` has much lower `id_hol_mse` than `shared_cfm` and `flat_pifm`.
2. `holonomy_connection` has `id_hol_pred_norm` close to `id_hol_true_norm`.
3. `holonomy_connection` has high `id_hol_cosine`, ideally > 0.7 in fast runs and > 0.85 in longer runs.
4. `flat_pifm` suppresses curvature and underpredicts loop residuals on the nonlinear world.
5. On `world=commute`, `holonomy_connection` does not hallucinate large loop residuals.

The strongest paper figure is a table like:

| Model | Local MSE | AB/BA MSE | Holonomy MSE | OOD Holonomy MSE | Curvature norm |
|---|---:|---:|---:|---:|---:|
| independent CFM | low/medium | high | high | high | uncontrolled |
| shared CFM | low | medium | medium | high | uncontrolled |
| flat PiFM | low | high | high | high | low |
| local connection | low | lower | medium | medium | medium |
| **holonomy connection** | low | low | **low** | **low** | nonzero |

---

## 10. Paper framing

One-sentence contribution:

> We introduce finite-time holonomy matching for controlled generative flow maps, targeting order-sensitive systems where nonzero commutator structure is signal rather than a path-independence defect.

This repo is meant to support an initial workshop paper. The synthetic experiment is designed to be lightweight but reviewer-legible: it includes a true noncommuting world, a commuting negative control, local CFM baselines, a flat/path-independent baseline, and finite-time loop metrics.

---

## Initial shipped results

I included quick CPU-scale results in:

```text
results/initial/nonlinear_quick_comparison.csv
results/initial/commute_quick_comparison.csv
results/initial/holonomy_scaling.csv
results/initial/flat_pifm_scaling.csv
docs/INITIAL_RESULTS.md
```

Reproduce them with:

```bash
DEVICE=auto bash scripts/reproduce_shipped_initial.sh
```

The high-level initial signal is: on the noncommuting nonlinear connection, `holonomy_connection` gives much lower finite-time commutator and loop-holonomy error than `shared_cfm`, `local_connection`, `independent_cfm`, and the flat/PiFM-style foil; on the commuting negative control it keeps loop residuals near zero. Treat these as sanity numbers, not final paper numbers.

---

---

## 9. A800 / CUDA runs

### Synthetic benchmark on two A800s

This launches the five synthetic baselines across two GPUs using CUDA + BF16 autocast:

```bash
STEPS=10000 BATCH=4096 HIDDEN=256 DEPTH=5 bash scripts/run_a800_synthetic_two_gpu.sh
```

Multi-seed run for paper tables:

```bash
SEEDS="0 1 2" STEPS=10000 BATCH=4096 bash scripts/run_a800_multiseed_synthetic.sh
```

Outputs:

```text
runs/a800_synthetic_nonlinear_seed0/comparison.csv
runs/a800_synthetic_nonlinear_all_seeds.csv
```

### Torchvision affine-MNIST visual benchmark

Dataset download:

```bash
python -m holoflow_conn.affine_mnist --data-dir data --download-only
```

Train four image baselines across two GPUs:

```bash
STEPS=8000 BATCH=1024 HIDDEN=96 DEPTH=6 bash scripts/run_a800_affine_mnist_two_gpu.sh
```

Outputs:

```text
runs/a800_affine_mnist_seed0/comparison.csv
runs/a800_affine_mnist_seed0/<model>/metrics.csv
runs/a800_affine_mnist_seed0/<model>/viz_step_8000.png
```

The affine-MNIST experiment is not meant to be SOTA image generation. It is a visual controlled-transform benchmark: rotate-then-translate differs from translate-then-rotate, and the method is evaluated on endpoint error, AB/BA commutator error, loop holonomy error, and false identity residual.

Suggested initial A800 settings:

```bash
# quick sanity
STEPS=1000 BATCH=512 bash scripts/run_a800_affine_mnist_two_gpu.sh

# more credible workshop-scale visual run
STEPS=8000 BATCH=1024 HIDDEN=96 DEPTH=6 bash scripts/run_a800_affine_mnist_two_gpu.sh
```

For full paper-ish numbers, run seeds 0/1/2 and report mean ± std.
