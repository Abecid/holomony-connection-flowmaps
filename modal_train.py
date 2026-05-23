"""Run the holonomy-connection-flowmaps training jobs on Modal.

Expected Modal resources are created lazily if they do not already exist:
- Volume `holonomy-flowmaps-runs` for checkpoints, logs, CSVs, and W&B files.
- Volume `holonomy-flowmaps-data` for affine-MNIST data.
- Volume `holonomy-flowmaps-cache` for torch/matplotlib/cache directories.
- Secret `wandb-adamlee00` with `WANDB_API_KEY` for online W&B.

Example:
    MODAL_PROFILE=hao-ai-lab modal run modal_train.py --mode synthetic --seeds 0
"""

from __future__ import annotations

import os

import modal

APP_NAME = "holonomy-connection-flowmaps-train"
PROJECT_DIR = "/root/holonomy-connection-flowmaps"

RUNS_MOUNT = "/runs"
DATA_MOUNT = "/data"
CACHE_MOUNT = "/cache"

RUNS_VOLUME_NAME = "holonomy-flowmaps-runs"
DATA_VOLUME_NAME = "holonomy-flowmaps-data"
CACHE_VOLUME_NAME = "holonomy-flowmaps-cache"
WANDB_SECRET_NAME = "wandb-adamlee00"
WANDB_ENTITY = "adamlee00"
WANDB_PROJECT = "holonomy-connection-flowmaps"


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        out = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return out


# Modal evaluates this when the app is loaded. Use e.g.
# MODAL_GPU_COUNT=4 modal run modal_train.py --mode synthetic --seeds 0
MODAL_GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "A100")
MODAL_GPU_COUNT = _positive_int_env("MODAL_GPU_COUNT", 2)
GPU = MODAL_GPU_TYPE if MODAL_GPU_COUNT == 1 else f"{MODAL_GPU_TYPE}:{MODAL_GPU_COUNT}"

runs_vol = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
cache_vol = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.micromamba(python_version="3.11")
    .apt_install("bash", "git", "build-essential")
    .micromamba_install(
        "pip",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
        "pytest",
        "wandb>=0.16",
        "pytorch",
        "torchvision",
        "pytorch-cuda=12.1",
        channels=["pytorch", "nvidia", "conda-forge"],
    )
    .add_local_dir(
        ".",
        PROJECT_DIR,
        copy=True,
        ignore=[
            ".git/**",
            ".venv/**",
            "__pycache__/**",
            ".pytest_cache/**",
            "data/**",
            "runs/**",
            "wandb/**",
        ],
    )
    .run_commands(
        f"cd {PROJECT_DIR} && python -m pip install -e .",
        (
            "python -c 'import torch, torchvision; "
            "print(torch.__version__, torchvision.__version__)'"
        ),
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_HOME": f"{CACHE_MOUNT}/torch",
            "XDG_CACHE_HOME": f"{CACHE_MOUNT}/xdg",
            "MPLCONFIGDIR": f"{CACHE_MOUNT}/matplotlib",
            "WANDB_DIR": f"{RUNS_MOUNT}/wandb",
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_ENTITY": WANDB_ENTITY,
        }
    )
)

app = modal.App(APP_NAME)


def _parse_seeds(seeds: str) -> list[int]:
    out = []
    for part in seeds.replace(",", " ").split():
        out.append(int(part))
    if not out:
        raise ValueError("Pass at least one seed, e.g. --seeds 0 or --seeds '0 1 2'.")
    return out


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * 60,
    volumes={
        RUNS_MOUNT: runs_vol,
        DATA_MOUNT: data_vol,
        CACHE_MOUNT: cache_vol,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def train(
    mode: str,
    seeds: str,
    world: str,
    steps: int,
    batch: int,
    hidden: int,
    depth: int,
    test_every: int,
    eval_batch: int,
    eval_batches: int,
    workers: int,
    wandb: bool,
    wandb_mode: str,
    wandb_artifacts: bool,
    run_id: str,
) -> str:
    from datetime import datetime, timezone
    import os
    from pathlib import Path
    import subprocess

    def run(cmd: list[str], *, env: dict[str, str]) -> None:
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_DIR, env={**os.environ, **env}, check=True)

    def commit_volumes() -> None:
        runs_vol.commit()
        data_vol.commit()
        cache_vol.commit()

    def preflight() -> int:
        import torch
        import torchvision

        print("=== Modal preflight ===", flush=True)
        print(f"torch={torch.__version__}", flush=True)
        print(f"torchvision={torchvision.__version__}", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the Modal container.")
        cuda_devices = torch.cuda.device_count()
        print(f"cuda_devices={cuda_devices}", flush=True)
        for idx in range(cuda_devices):
            print(f"cuda:{idx} {torch.cuda.get_device_name(idx)}", flush=True)
        subprocess.run(["python", "-m", "pytest", "tests/test_integrators.py", "tests/test_models.py", "-q"], cwd=PROJECT_DIR, check=True)
        print("=== Modal preflight OK ===", flush=True)
        return cuda_devices

    def base_env(seed: int, out: str) -> dict[str, str]:
        return {
            "SEED": str(seed),
            "OUT": out,
            "STEPS": str(steps),
            "BATCH": str(batch),
            "HIDDEN": str(hidden),
            "DEPTH": str(depth),
            "TEST_EVERY": str(test_every),
            "EVAL_BATCH": str(eval_batch),
            "EVAL_BATCHES": str(eval_batches),
            "WANDB": "1" if wandb else "0",
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_GROUP": Path(out).name,
            "WANDB_MODE": wandb_mode,
            "WANDB_ARTIFACTS": "1" if wandb_artifacts else "0",
            "NUM_GPUS": str(cuda_devices),
        }

    def combine_synthetic(seed_dirs: list[Path], out: Path) -> None:
        import pandas as pd

        rows = []
        for seed_dir in seed_dirs:
            comparison = seed_dir / "comparison.csv"
            if comparison.exists():
                df = pd.read_csv(comparison)
                df["seed_dir"] = seed_dir.name
                rows.append(df)
        if rows:
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(rows, ignore_index=True).to_csv(out, index=False)
            print(f"Wrote {out}", flush=True)

    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    mode = mode.strip().lower().replace("-", "_")
    seed_values = _parse_seeds(seeds)

    cuda_devices = preflight()

    if mode == "synthetic":
        seed_dirs: list[Path] = []
        for seed in seed_values:
            out = f"{RUNS_MOUNT}/synthetic_{world}_{run_id}_seed{seed}"
            env = {**base_env(seed, out), "WORLD": world}
            run(["bash", "scripts/run_a800_synthetic_two_gpu.sh"], env=env)
            seed_dirs.append(Path(out))
            commit_volumes()
        if len(seed_dirs) > 1:
            combine_synthetic(
                seed_dirs,
                Path(RUNS_MOUNT) / f"synthetic_{world}_{run_id}_all_seeds.csv",
            )
            commit_volumes()
        return str(seed_dirs[-1])

    if mode == "commute":
        seed_dirs = []
        for seed in seed_values:
            out = f"{RUNS_MOUNT}/synthetic_commute_{run_id}_seed{seed}"
            env = {**base_env(seed, out), "WORLD": "commute"}
            run(["bash", "scripts/run_a800_synthetic_two_gpu.sh"], env=env)
            seed_dirs.append(Path(out))
            commit_volumes()
        if len(seed_dirs) > 1:
            combine_synthetic(
                seed_dirs,
                Path(RUNS_MOUNT) / f"synthetic_commute_{run_id}_all_seeds.csv",
            )
            commit_volumes()
        return str(seed_dirs[-1])

    if mode == "affine_mnist":
        seed_dirs = []
        for seed in seed_values:
            out = f"{RUNS_MOUNT}/affine_mnist_{run_id}_seed{seed}"
            env = {
                **base_env(seed, out),
                "DATA": f"{DATA_MOUNT}/mnist",
                "WORKERS": str(workers),
            }
            run(["bash", "scripts/run_a800_affine_mnist_two_gpu.sh"], env=env)
            seed_dirs.append(Path(out))
            commit_volumes()
        return str(seed_dirs[-1])

    raise ValueError("mode must be one of: synthetic, commute, affine_mnist")


@app.local_entrypoint()
def main(
    mode: str = "synthetic",
    seeds: str = "0",
    world: str = "nonlinear",
    steps: int = 0,
    batch: int = 0,
    hidden: int = 0,
    depth: int = 0,
    test_every: int = 500,
    eval_batch: int = 0,
    eval_batches: int = 2,
    workers: int = 8,
    wandb: bool = True,
    wandb_mode: str = "online",
    wandb_artifacts: bool = False,
    run_id: str = "",
) -> None:
    """Launch a README-aligned Modal training job."""

    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode == "affine_mnist":
        steps = steps or 8000
        batch = batch or 1024
        hidden = hidden or 96
        depth = depth or 6
        eval_batch = eval_batch or batch
    else:
        steps = steps or 10000
        batch = batch or 4096
        hidden = hidden or 256
        depth = depth or 5
        eval_batch = eval_batch or 4096

    out = train.remote(
        normalized_mode,
        seeds,
        world,
        steps,
        batch,
        hidden,
        depth,
        test_every,
        eval_batch,
        eval_batches,
        workers,
        wandb,
        wandb_mode,
        wandb_artifacts,
        run_id,
    )
    print(f"Modal run finished. Latest output dir: {out}")
