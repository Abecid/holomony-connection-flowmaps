from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def add_wandb_args(parser: Any) -> None:
    """Attach common Weights & Biases CLI args to an argparse parser."""
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="holonomy-connection-flowmaps", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/team.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-name", dest="wandb_run_name", type=str, default=None, help="Alias for --wandb-run-name.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group, useful for baseline/multi-seed sweeps.")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None, help="Optional W&B mode override.")
    parser.add_argument("--wandb-log-artifacts", action="store_true", help="Log config/checkpoint/metrics/viz files as W&B artifacts.")
    parser.add_argument("--wandb-watch", action="store_true", help="Call wandb.watch(model). Can add overhead.")


def _tags(tags: str) -> list[str]:
    return [t.strip() for t in tags.split(",") if t.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _jsonable(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def init_wandb(args: Any, *, config: dict[str, Any], device: Any | None = None, run_kind: str = "train", model: Any | None = None):
    """Initialize W&B and return the wandb module, or None if disabled.

    Returning the module lets callers use `wandb_run.log(...)`,
    `wandb_run.define_metric(...)`, and `wandb_run.finish()` uniformly.
    """
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "--wandb was passed, but the 'wandb' package is not installed. "
            "Install with `pip install wandb` or recreate the environment from environment-cuda.yml."
        ) from exc

    cfg = _jsonable(config)
    cfg["run_kind"] = run_kind
    if device is not None:
        cfg["device"] = str(device)
    init_kwargs: dict[str, Any] = {
        "project": getattr(args, "wandb_project", "holonomy-connection-flowmaps"),
        "config": cfg,
        "tags": _tags(getattr(args, "wandb_tags", "")),
    }
    if getattr(args, "wandb_entity", None):
        init_kwargs["entity"] = args.wandb_entity
    if getattr(args, "wandb_run_name", None):
        init_kwargs["name"] = args.wandb_run_name
    if getattr(args, "wandb_group", None):
        init_kwargs["group"] = args.wandb_group
    if getattr(args, "wandb_mode", None):
        init_kwargs["mode"] = args.wandb_mode
    wandb.init(**init_kwargs)
    if model is not None and getattr(args, "wandb_watch", False):
        wandb.watch(model, log="gradients", log_freq=100)
    return wandb


def paper_metric_aliases(evals: dict[str, Any]) -> dict[str, float]:
    """Stable `paper/*` aliases for the metrics that validate the method.

    Synthetic trainer names include `id_hol_*`; affine-MNIST uses `hol_*`.
    This maps both onto a paper-facing vocabulary so W&B panels stay stable.
    """
    mapping = {
        "paper/local_mse": "local_mse",
        "paper/mse_ab_ba": "ab_ba_endpoint_mse",
        "paper/mse_random": "random_endpoint_mse",
        "paper/comm_mse": "ab_ba_comm_mse",
        "paper/comm_cosine": "ab_ba_comm_cosine",
        "paper/comm_true_norm": "ab_ba_comm_true_norm",
        "paper/comm_pred_norm": "ab_ba_comm_pred_norm",
        "paper/mse_loops": "id_hol_mse",
        "paper/loop_holonomy_mse": "id_hol_mse",
        "paper/loop_holonomy_cosine": "id_hol_cosine",
        "paper/loop_true_norm": "id_hol_true_norm",
        "paper/loop_pred_norm": "id_hol_pred_norm",
        "paper/loop_pred_identity_mse": "id_false_identity_mse",
        "paper/ood_comm_mse": "ood_ab_ba_comm_mse",
        "paper/ood_comm_cosine": "ood_ab_ba_comm_cosine",
        "paper/ood_loop_holonomy_mse": "ood_hol_mse",
        "paper/ood_loop_holonomy_cosine": "ood_hol_cosine",
        "paper/ood_loop_pred_identity_mse": "ood_false_identity_mse",
        "paper/flatness_norm2": "flatness_norm2",
    }
    # Affine-MNIST equivalent keys.
    affine_mapping = {
        "paper/mse_loops": "hol_mse",
        "paper/loop_holonomy_mse": "hol_mse",
        "paper/loop_holonomy_cosine": "hol_cosine",
        "paper/loop_true_norm": "hol_true_norm",
        "paper/loop_pred_norm": "hol_pred_norm",
        "paper/loop_pred_identity_mse": "false_identity_mse",
        "paper/ood_loop_holonomy_mse": "ood_hol_mse",
        "paper/ood_loop_holonomy_cosine": "ood_hol_cosine",
        "paper/ood_loop_pred_identity_mse": "ood_false_identity_mse",
        "paper/flatness_norm2": "flatness_norm2",
    }
    out: dict[str, float] = {}
    for alias, key in mapping.items():
        val = evals.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            out[alias] = float(val)
    for alias, key in affine_mapping.items():
        if alias in out:
            continue
        val = evals.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            out[alias] = float(val)
    return out


class BestTracker:
    """Track best-so-far paper metrics and return W&B-ready logs.

    Lower is better except cosine-like metrics, where higher is better.
    """

    def __init__(self) -> None:
        self.best: dict[str, tuple[float, int]] = {}

    @staticmethod
    def maximize(key: str) -> bool:
        return key.endswith("cosine") or key.endswith("accuracy")

    def update(self, metrics: dict[str, Any], step: int) -> tuple[dict[str, float | int], list[str]]:
        logs: dict[str, float | int] = {}
        improved: list[str] = []
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            prev = self.best.get(key)
            better = prev is None or (value_f > prev[0] if self.maximize(key) else value_f < prev[0])
            if better:
                self.best[key] = (value_f, int(step))
                improved.append(key)
            best_value, best_step = self.best[key]
            alias = key.split("paper/", 1)[-1] if key.startswith("paper/") else key.replace("/", "_")
            logs[f"best/{alias}"] = best_value
            logs[f"best/{alias}_step"] = best_step
        return logs, improved


def wandb_log_artifacts(wandb_run: Any, outdir: str | Path, *, name: str, aliases: list[str] | None = None) -> None:
    """Log useful run outputs as a single W&B artifact directory."""
    out = Path(outdir)
    if not out.exists():
        return
    safe_name = name.replace("/", "-").replace(" ", "_")
    artifact = wandb_run.Artifact(safe_name, type="run-output")
    for fname in ["config.json", "metrics.csv", "eval_metrics.csv", "holonomy_scaling.csv", "checkpoint.pt"]:
        p = out / fname
        if p.exists():
            artifact.add_file(str(p))
    for p in sorted(out.glob("best_*.pt")):
        artifact.add_file(str(p))
    for p in sorted(list(out.glob("viz_step_*.png")) + list(out.glob("*.png")))[-6:]:
        artifact.add_file(str(p))
    if aliases is None:
        aliases = ["latest"]
    wandb_run.log_artifact(artifact, aliases=aliases)
