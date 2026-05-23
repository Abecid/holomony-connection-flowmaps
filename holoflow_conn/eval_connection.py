from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch

from .metrics import evaluate_model, holonomy_scaling
from .models import VFConfig, build_model
from .utils import append_csv, get_device, save_json, seed_all
from .worlds import WorldConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a controlled-connection checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--model-steps", type=int, default=None)
    p.add_argument("--gt-steps", type=int, default=32)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--scaling", action="store_true")
    p.add_argument("--deltas", type=str, default="0.05,0.1,0.2,0.4,0.8,1.0")
    return p.parse_args()


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model_cfg = VFConfig(**ckpt["model_cfg"])
    world_cfg = WorldConfig(**ckpt["world_cfg"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_cfg, world_cfg, ckpt


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = get_device(args.device)
    model, model_cfg, world_cfg, ckpt = load_checkpoint(args.checkpoint, device)
    outdir = Path(args.outdir) if args.outdir else Path(args.checkpoint).parent
    outdir.mkdir(parents=True, exist_ok=True)
    model_steps = args.model_steps or ckpt.get("args", {}).get("model_steps", 8)
    train_low = ckpt.get("args", {}).get("train_low", 0.15)
    train_high = ckpt.get("args", {}).get("train_high", 0.75)

    metrics = evaluate_model(
        model,
        world_cfg,
        device,
        batch_size=args.batch_size,
        batches=args.batches,
        train_low=train_low,
        train_high=train_high,
        n_steps_model=model_steps,
        n_steps_gt=args.gt_steps,
    )
    metrics = {"model": model_cfg.model, "world": world_cfg.world, **metrics}
    save_json(metrics, outdir / "eval_metrics.json")
    pd.DataFrame([metrics]).to_csv(outdir / "eval_metrics.csv", index=False)
    print(pd.DataFrame([metrics]).T.to_string(header=False))

    if args.scaling:
        deltas = [float(x) for x in args.deltas.split(",") if x]
        rows = holonomy_scaling(model, world_cfg, device, deltas=deltas, batch=args.batch_size, n_steps_model=model_steps, n_steps_gt=args.gt_steps)
        for r in rows:
            r["model"] = model_cfg.model
            r["world"] = world_cfg.world
        pd.DataFrame(rows).to_csv(outdir / "holonomy_scaling.csv", index=False)
        print("\nHolonomy scaling:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
