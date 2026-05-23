# GPU/A800 Experiment Plan

The project has two tiers.

## Tier 1: synthetic controlled connection

Purpose: validate the method in a system with known vector fields and known holonomy.

Run:

```bash
conda env create -f environment-cuda.yml
conda activate holonomy-connection-flowmaps
pip install -e .
STEPS=10000 BATCH=4096 HIDDEN=256 DEPTH=5 bash scripts/run_a800_synthetic_two_gpu.sh
```

Paper table metrics:

- `ab_ba_comm_mse` ↓
- `ab_ba_comm_cosine` ↑
- `id_hol_mse` ↓
- `id_hol_cosine` ↑
- `ood_hol_mse` ↓
- `flatness_norm2` as diagnostic

Expected qualitative pattern:

- `flat_pifm` should have low curvature but high noncommuting-world holonomy error.
- `shared_cfm` should learn local velocities but underperform on finite-time loops.
- `holonomy_connection` should best match AB/BA commutators and loop residuals.

## Tier 2: torchvision affine-MNIST

Purpose: demonstrate the same order-sensitive effect on real images, cheaply.

Run:

```bash
python -m holoflow_conn.affine_mnist --data-dir data --download-only
STEPS=8000 BATCH=1024 HIDDEN=96 DEPTH=6 bash scripts/run_a800_affine_mnist_two_gpu.sh
```

Output images:

```text
runs/a800_affine_mnist_seed0/<model>/viz_step_8000.png
```

Each visualization row block is:

1. original image
2. true AB
3. predicted AB
4. true BA
5. predicted BA

This is a sanity/visual benchmark, not a generative SOTA benchmark.

## Why parallel scripts instead of DDP?

For the synthetic benchmark, each run is a small MLP with lots of Python-level rollout structure. DDP would mostly synchronize gradients for a model that is already small. Running independent models/seeds across GPU 0/1 is higher-throughput and gives the result table faster.

For affine-MNIST, one process per model per GPU is also simpler. If the CNN is scaled up substantially, DDP can be added later, but it is not the right first bottleneck.
