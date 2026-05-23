from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch
from tqdm import trange

from .data import sample_cycle_batch, sample_holonomy_batch, sample_local_batch, sample_rollout_batch
from .losses import LossWeights, cycle_loss, default_weights, flatness_penalty, holonomy_loss, local_velocity_loss, rollout_loss
from .metrics import evaluate_model
from .models import VFConfig, build_model
from .gpu import autocast_context, enable_fast_cuda, make_grad_scaler
from .utils import append_csv, atomic_torch_save, count_parameters, get_device, save_json, seed_all
from .wandb_utils import BestTracker, add_wandb_args, init_wandb, paper_metric_aliases, wandb_log_artifacts
from .worlds import WorldConfig, sample_base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train controlled-connection flow maps with holonomy matching.")
    p.add_argument("--model", choices=["independent_cfm", "shared_cfm", "local_connection", "flat_pifm", "holonomy_connection"], default="holonomy_connection")
    p.add_argument("--world", choices=["se2", "body", "nonlinear", "commute"], default="nonlinear")
    p.add_argument("--outdir", type=str, default="runs/holonomy_connection")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--flat-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--act", choices=["silu", "gelu", "relu", "tanh"], default="silu")
    p.add_argument("--train-low", type=float, default=0.15)
    p.add_argument("--train-high", type=float, default=0.75)
    p.add_argument("--rollout-family", choices=["ab_ba", "loops", "random", "mixed"], default="mixed")
    p.add_argument("--path-len", type=int, default=4)
    p.add_argument("--model-steps", type=int, default=6, help="RK4 substeps used for differentiable model rollouts during training.")
    p.add_argument("--gt-steps", type=int, default=16, help="RK4 substeps used for ground-truth trajectory generation.")
    p.add_argument("--eval-every", type=int, default=500, help="Run held-out paper metric suite every N training steps.")
    p.add_argument("--test-every", type=int, default=None, help="Alias for --eval-every; useful when thinking of held-out test metrics.")
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-gt-steps", type=int, default=24)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use autocast. On A800, default dtype bf16 is recommended.")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable TF32 matmuls on Ampere GPUs.")
    p.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rot-scale", type=float, default=1.0)
    p.add_argument("--trans-scale", type=float, default=0.75)
    p.add_argument("--shear-scale", type=float, default=0.35)
    p.add_argument("--nonlinear-scale", type=float, default=0.25)
    p.add_argument("--local-weight", type=float, default=None)
    p.add_argument("--rollout-weight", type=float, default=None)
    p.add_argument("--holonomy-weight", type=float, default=None)
    p.add_argument("--cycle-weight", type=float, default=None)
    p.add_argument("--flat-weight", type=float, default=None)
    p.add_argument("--save-best-checkpoints", action=argparse.BooleanOptionalAction, default=True, help="Save best_* checkpoints for key held-out paper metrics.")
    p.add_argument("--log-timing", action=argparse.BooleanOptionalAction, default=True, help="Log coarse train/eval timing diagnostics.")
    add_wandb_args(p)
    args = p.parse_args()
    if args.test_every is not None:
        args.eval_every = args.test_every
    return args


def resolve_weights(args: argparse.Namespace) -> LossWeights:
    w = default_weights(args.model)
    if args.local_weight is not None:
        w.local = args.local_weight
    if args.rollout_weight is not None:
        w.rollout = args.rollout_weight
    if args.holonomy_weight is not None:
        w.holonomy = args.holonomy_weight
    if args.cycle_weight is not None:
        w.cycle = args.cycle_weight
    if args.flat_weight is not None:
        w.flat = args.flat_weight
    return w


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    enable_fast_cuda(tf32=args.tf32, benchmark=args.cudnn_benchmark)
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = get_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = WorldConfig(
        world=args.world,
        rot_scale=args.rot_scale,
        trans_scale=args.trans_scale,
        shear_scale=args.shear_scale,
        nonlinear_scale=args.nonlinear_scale,
    )
    model_cfg = VFConfig(model=args.model, hidden=args.hidden, depth=args.depth, act=args.act)
    model = build_model(model_cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    weights = resolve_weights(args)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, enabled=args.amp, dtype=args.amp_dtype)
    run_config = {
        "args": vars(args),
        "world_cfg": cfg,
        "model_cfg": model_cfg,
        "weights": weights,
        "num_parameters": count_parameters(model),
    }
    save_json(run_config, outdir / "config.json")
    wandb_run = init_wandb(args, config=run_config, device=device, run_kind="synthetic_connection_train", model=model)
    if wandb_run is not None:
        wandb_run.define_metric("train/step")
        wandb_run.define_metric("train/*", step_metric="train/step")
        wandb_run.define_metric("test/*", step_metric="train/step")
        wandb_run.define_metric("paper/*", step_metric="train/step")
        wandb_run.define_metric("best/*", step_metric="train/step")

    best_tracker = BestTracker()
    metrics_path = outdir / "metrics.csv"
    pbar = trange(1, args.steps + 1, desc=f"{args.model}/{args.world}", mininterval=1.0)
    last_row: dict[str, float] = {}
    last_timing_step = 0
    last_timing_time = time.perf_counter()
    for step in pbar:
        step_start = time.perf_counter()
        model.train()
        loss_terms: dict[str, torch.Tensor] = {}

        with autocast_context(device, enabled=args.amp, dtype=args.amp_dtype):
            if weights.local:
                lb = sample_local_batch(args.batch_size, cfg, device)
                loss_terms["local_loss"] = local_velocity_loss(model, lb)
            if weights.rollout:
                rb = sample_rollout_batch(
                    args.batch_size,
                    cfg,
                    device,
                    family=args.rollout_family,
                    amount_low=args.train_low,
                    amount_high=args.train_high,
                    signed=True,
                    path_len=args.path_len,
                    gt_steps=args.gt_steps,
                )
                loss_terms["rollout_loss"] = rollout_loss(model, rb.x0, rb.c0, rb.dirs, rb.amounts, rb.y, n_steps=args.model_steps)
            if weights.holonomy:
                hb = sample_holonomy_batch(args.batch_size, cfg, device, delta_low=args.train_low, delta_high=args.train_high, gt_steps=args.gt_steps)
                loss_terms["holonomy_loss"] = holonomy_loss(model, hb.x0, hb.c0, hb.dirs, hb.amounts, hb.true_residual, n_steps=args.model_steps)
            if weights.cycle:
                x0, c0, dirs, amounts = sample_cycle_batch(args.batch_size, device, delta_low=args.train_low, delta_high=args.train_high)
                loss_terms["cycle_loss"] = cycle_loss(model, x0, c0, dirs, amounts, n_steps=args.model_steps)

            loss = torch.zeros((), device=device)
            loss = loss + weights.local * loss_terms.get("local_loss", torch.zeros((), device=device))
            loss = loss + weights.rollout * loss_terms.get("rollout_loss", torch.zeros((), device=device))
            loss = loss + weights.holonomy * loss_terms.get("holonomy_loss", torch.zeros((), device=device))
            loss = loss + weights.cycle * loss_terms.get("cycle_loss", torch.zeros((), device=device))

        # Higher-order flatness penalty is intentionally kept in fp32; AMP + create_graph
        # can be numerically noisy and is not the bottleneck if flat_batch_size is modest.
        if weights.flat:
            x_flat = sample_base(args.flat_batch_size, device)
            c_flat = -0.6 + 1.2 * torch.rand(args.flat_batch_size, 2, device=device)
            loss_terms["flat_loss"] = flatness_penalty(model, x_flat, c_flat)
            loss = loss + weights.flat * loss_terms["flat_loss"]

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 25 == 0:
            last_row = {k: float(v.detach().cpu()) for k, v in loss_terms.items()}
            last_row["loss"] = float(loss.detach().cpu())
            last_row["grad_norm"] = float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm)
            last_row["lr"] = float(opt.param_groups[0]["lr"])
            if args.log_timing:
                now = time.perf_counter()
                interval_steps = step - last_timing_step
                if interval_steps > 0:
                    last_row["sec_per_step"] = (now - last_timing_time) / interval_steps
                last_row["last_step_sec"] = now - step_start
                last_timing_step = step
                last_timing_time = now
            pbar.set_postfix(loss=f"{last_row['loss']:.3e}", loc=f"{last_row.get('local_loss', 0):.2e}", hol=f"{last_row.get('holonomy_loss', 0):.2e}")
            if wandb_run is not None:
                wandb_run.log({"train/step": step, **{f"train/{k}": v for k, v in last_row.items()}}, step=step)

        if step == args.steps or (args.eval_every and step % args.eval_every == 0):
            eval_start = time.perf_counter()
            evals = evaluate_model(
                model,
                cfg,
                device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                train_low=args.train_low,
                train_high=args.train_high,
                n_steps_model=args.model_steps,
                n_steps_gt=args.eval_gt_steps,
            )
            if args.log_timing:
                evals["eval_sec"] = time.perf_counter() - eval_start
            paper_metrics = paper_metric_aliases(evals)
            best_source = dict(paper_metrics)
            # Identity residual is a negative-control metric: lower is good only in the
            # commuting/flat world. In noncommuting worlds, selecting for low identity
            # would wrongly prefer models that suppress true holonomy.
            if args.world != "commute":
                best_source.pop("paper/loop_pred_identity_mse", None)
                best_source.pop("paper/ood_loop_pred_identity_mse", None)
            best_logs, improved = best_tracker.update(best_source, step)
            row: dict[str, Any] = {"step": step, **last_row, **evals, **{k.replace("/", "_"): v for k, v in paper_metrics.items()}, **{k.replace("/", "_"): v for k, v in best_logs.items()}}
            append_csv(row, metrics_path)
            checkpoint = {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "world_cfg": cfg.__dict__,
                "weights": weights.__dict__,
                "args": vars(args),
                "step": step,
                "evals": evals,
                "paper_metrics": paper_metrics,
            }
            atomic_torch_save(checkpoint, outdir / "checkpoint.pt")
            if args.save_best_checkpoints:
                important = {
                    "paper/comm_mse": "best_comm_mse.pt",
                    "paper/comm_cosine": "best_comm_cosine.pt",
                    "paper/loop_holonomy_mse": "best_loop_holonomy_mse.pt",
                    "paper/ood_loop_holonomy_mse": "best_ood_loop_holonomy_mse.pt",
                }
                if args.world == "commute":
                    important["paper/loop_pred_identity_mse"] = "best_loop_pred_identity_mse.pt"
                    important["paper/ood_loop_pred_identity_mse"] = "best_ood_loop_pred_identity_mse.pt"
                for key in improved:
                    if key in important:
                        atomic_torch_save(checkpoint, outdir / important[key])
            if wandb_run is not None:
                wandb_run.log({
                    "train/step": step,
                    **{f"test/{k}": float(v) for k, v in evals.items()},
                    **paper_metrics,
                    **best_logs,
                }, step=step)
    if wandb_run is not None:
        if args.wandb_log_artifacts:
            wandb_log_artifacts(wandb_run, outdir, name=f"{args.model}-{args.world}-seed{args.seed}", aliases=["latest", f"step-{args.steps}"])
        wandb_run.finish()
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
