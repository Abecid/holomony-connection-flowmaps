from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import sample_cycle_batch, sample_holonomy_batch, sample_local_batch, sample_rollout_batch
from .losses import cycle_loss, default_weights, flatness_penalty, holonomy_loss, local_velocity_loss, rollout_loss
from .metrics import evaluate_model
from .models import VFConfig, build_model
from .utils import get_device, seed_all
from .worlds import WorldConfig, sample_base


def parse_args():
    p = argparse.ArgumentParser(description="Run all baselines in one Python process for quick initial numbers.")
    p.add_argument("--world", choices=["se2", "body", "nonlinear", "commute"], default="nonlinear")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--model-steps", type=int, default=3)
    p.add_argument("--gt-steps", type=int, default=6)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-batches", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="runs/quick_benchmark")
    p.add_argument("--num-threads", type=int, default=1)
    return p.parse_args()


def train_one(model_name: str, cfg: WorldConfig, args, device: torch.device) -> tuple[torch.nn.Module, dict[str, float]]:
    model = build_model(VFConfig(model=model_name, hidden=args.hidden, depth=args.depth)).to(device)
    weights = default_weights(model_name)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    logs = {}
    for _ in range(args.steps):
        loss = torch.zeros((), device=device)
        if weights.local:
            loc = local_velocity_loss(model, sample_local_batch(args.batch_size, cfg, device))
            loss = loss + weights.local * loc
            logs["local_loss"] = float(loc.detach().cpu())
        if weights.rollout:
            rb = sample_rollout_batch(args.batch_size, cfg, device, family="mixed", amount_low=0.15, amount_high=0.75, gt_steps=args.gt_steps)
            rol = rollout_loss(model, rb.x0, rb.c0, rb.dirs, rb.amounts, rb.y, n_steps=args.model_steps)
            loss = loss + weights.rollout * rol
            logs["rollout_loss"] = float(rol.detach().cpu())
        if weights.holonomy:
            hb = sample_holonomy_batch(args.batch_size, cfg, device, delta_low=0.15, delta_high=0.75, gt_steps=args.gt_steps)
            hol = holonomy_loss(model, hb.x0, hb.c0, hb.dirs, hb.amounts, hb.true_residual, n_steps=args.model_steps)
            loss = loss + weights.holonomy * hol
            logs["holonomy_loss"] = float(hol.detach().cpu())
        if weights.cycle:
            x0, c0, dirs, amounts = sample_cycle_batch(args.batch_size, device)
            cyc = cycle_loss(model, x0, c0, dirs, amounts, n_steps=args.model_steps)
            loss = loss + weights.cycle * cyc
            logs["cycle_loss"] = float(cyc.detach().cpu())
        if weights.flat:
            xb = sample_base(min(64, args.batch_size), device)
            cb = -0.6 + 1.2 * torch.rand(xb.shape[0], 2, device=device)
            fl = flatness_penalty(model, xb, cb)
            loss = loss + weights.flat * fl
            logs["flat_loss"] = float(fl.detach().cpu())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    logs["final_loss"] = float(loss.detach().cpu())
    return model, logs


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    device = get_device(args.device)
    cfg = WorldConfig(world=args.world)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models = ["independent_cfm", "shared_cfm", "local_connection", "flat_pifm", "holonomy_connection"]
    rows = []
    for i, name in enumerate(models):
        seed_all(args.seed + i)
        print(f"training {name}...")
        model, logs = train_one(name, cfg, args, device)
        metrics = evaluate_model(
            model,
            cfg,
            device,
            batch_size=args.eval_batch_size,
            batches=args.eval_batches,
            n_steps_model=args.model_steps,
            n_steps_gt=max(12, args.gt_steps * 2),
        )
        rows.append({"model": name, **logs, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "comparison.csv", index=False)
    print(df[["model", "local_mse", "ab_ba_endpoint_mse", "ab_ba_comm_mse", "id_hol_mse", "id_hol_true_norm", "id_hol_pred_norm", "id_hol_cosine", "ood_hol_mse", "flatness_norm2"]].to_string(index=False))


if __name__ == "__main__":
    main()
