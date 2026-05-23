from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

try:
    from torchvision import datasets, transforms, utils as vutils
except Exception as exc:  # pragma: no cover
    datasets = None
    transforms = None
    vutils = None
    _TORCHVISION_ERR = exc
else:
    _TORCHVISION_ERR = None

from .gpu import autocast_context, enable_fast_cuda, make_grad_scaler
from .utils import append_csv, atomic_torch_save, count_parameters, get_device, save_json, seed_all
from .wandb_utils import BestTracker, add_wandb_args, init_wandb, paper_metric_aliases, wandb_log_artifacts


@dataclass
class AffineConfig:
    image_size: int = 28
    rot_deg_per_unit: float = 35.0
    translate_px_per_unit: float = 7.0
    align_corners: bool = False


def _require_torchvision() -> None:
    if datasets is None or transforms is None:
        raise RuntimeError(f"torchvision is required for affine_mnist.py; import error was: {_TORCHVISION_ERR}")


def affine_theta(batch: int, angle_rad: torch.Tensor, tx_px: torch.Tensor, cfg: AffineConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # affine_grid uses normalized translation in [-1, 1]. This matrix maps output
    # coordinates to input coordinates; sign conventions do not matter for the
    # experiment as long as they are consistent and noncommuting.
    angle_rad = angle_rad.to(device=device, dtype=dtype)
    tx_norm = (2.0 * tx_px.to(device=device, dtype=dtype)) / float(cfg.image_size)
    z = torch.zeros(batch, device=device, dtype=dtype)
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    theta = torch.stack(
        [torch.stack([c, -s, tx_norm], dim=-1), torch.stack([s, c, z], dim=-1)],
        dim=1,
    )
    return theta


def apply_affine_step(x: torch.Tensor, direction: int, amount: torch.Tensor, cfg: AffineConfig) -> torch.Tensor:
    B, C, H, W = x.shape
    amount = amount.to(device=x.device, dtype=x.dtype)
    if amount.ndim == 0:
        amount = amount.expand(B)
    if direction == 0:
        angle = amount * (math.pi / 180.0) * cfg.rot_deg_per_unit
        tx = torch.zeros(B, device=x.device, dtype=x.dtype)
    elif direction == 1:
        angle = torch.zeros(B, device=x.device, dtype=x.dtype)
        tx = amount * cfg.translate_px_per_unit
    else:
        raise ValueError(direction)
    theta = affine_theta(B, angle, tx, cfg, x.device, x.dtype)
    grid = F.affine_grid(theta, size=x.shape, align_corners=cfg.align_corners)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=cfg.align_corners)


def apply_path_teacher(x: torch.Tensor, dirs: torch.Tensor, amounts: torch.Tensor, cfg: AffineConfig) -> torch.Tensor:
    y = x
    L = dirs.shape[1]
    for j in range(L):
        d = dirs[:, j]
        amt = amounts[:, j]
        y_next = y.clone()
        ma = d == 0
        mb = d == 1
        if ma.any():
            y_next[ma] = apply_affine_step(y[ma], 0, amt[ma], cfg)
        if mb.any():
            y_next[mb] = apply_affine_step(y[mb], 1, amt[mb], cfg)
        y = y_next
    return y.clamp(0.0, 1.0)


def update_controls(c: torch.Tensor, direction: int, h: torch.Tensor) -> torch.Tensor:
    e = torch.zeros_like(c)
    e[:, direction] = 1.0
    return c + h[:, None] * e


class ConvVF(nn.Module):
    """Small CNN vector field for 28x28 controlled image dynamics.

    Inputs are image plus four conditioning maps: alpha, beta, dir_A, dir_B.
    Output has the same shape as the image and represents dx/dcontrol.
    """

    def __init__(self, hidden: int = 64, depth: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1 + 2 + 2
        ch = hidden
        layers += [nn.Conv2d(in_ch, ch, 3, padding=1), nn.SiLU()]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8 if ch >= 8 else 1, ch), nn.SiLU()]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def velocity(self, x: torch.Tensor, controls: torch.Tensor, direction: int) -> torch.Tensor:
        B, _, H, W = x.shape
        dtype = x.dtype
        ctrl = controls.to(dtype=dtype).view(B, 2, 1, 1).expand(B, 2, H, W)
        tok = torch.zeros(B, 2, H, W, device=x.device, dtype=dtype)
        tok[:, direction] = 1.0
        return self.net(torch.cat([x, ctrl, tok], dim=1))

    def forward(self, x: torch.Tensor, controls: torch.Tensor, direction: int) -> torch.Tensor:
        return self.velocity(x, controls, direction)


def integrate_direction(model: ConvVF, x: torch.Tensor, c: torch.Tensor, direction: int, amount: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    amount = amount.to(device=x.device, dtype=x.dtype)
    if amount.ndim == 0:
        amount = amount.expand(x.shape[0])
    h = amount / float(n_steps)
    y, cc = x, c
    for _ in range(n_steps):
        v = model.velocity(y, cc, direction)
        y = y + h[:, None, None, None] * v
        cc = update_controls(cc, direction, h.to(dtype=cc.dtype))
    return y, cc


def integrate_path_model(model: ConvVF, x: torch.Tensor, c: torch.Tensor, dirs: torch.Tensor, amounts: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    y, cc = x, c
    L = dirs.shape[1]
    for j in range(L):
        d = dirs[:, j]
        amt = amounts[:, j]
        y_next = y.clone()
        c_next = cc.clone()
        ma = d == 0
        mb = d == 1
        if ma.any():
            ya, ca = integrate_direction(model, y[ma], cc[ma], 0, amt[ma], n_steps=n_steps)
            y_next[ma], c_next[ma] = ya, ca
        if mb.any():
            yb, cb = integrate_direction(model, y[mb], cc[mb], 1, amt[mb], n_steps=n_steps)
            y_next[mb], c_next[mb] = yb, cb
        y, cc = y_next, c_next
    return y, cc


def make_ab_ba(batch: int, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dirs_ab = torch.tensor([[0, 1]], device=device, dtype=torch.long).repeat(batch, 1)
    dirs_ba = torch.tensor([[1, 0]], device=device, dtype=torch.long).repeat(batch, 1)
    return dirs_ab, torch.stack([a, b], dim=1), dirs_ba, torch.stack([b, a], dim=1)


def make_loop(batch: int, delta: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dirs = torch.tensor([[0, 1, 0, 1]], device=device, dtype=torch.long).repeat(batch, 1)
    amounts = torch.stack([delta, delta, -delta, -delta], dim=1)
    return dirs, amounts


def random_paths(batch: int, length: int, low: float, high: float, device: torch.device, signed: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    dirs = torch.randint(0, 2, (batch, length), device=device)
    amounts = low + (high - low) * torch.rand(batch, length, device=device)
    if signed:
        amounts = amounts * torch.where(torch.rand_like(amounts) < 0.5, -1.0, 1.0)
    return dirs, amounts


def mnist_loader(data_dir: str, batch_size: int, workers: int, train: bool, download: bool) -> DataLoader:
    _require_torchvision()
    ds = datasets.MNIST(data_dir, train=train, download=download, transform=transforms.ToTensor())
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def infinite_images(loader: DataLoader, device: torch.device) -> Iterator[torch.Tensor]:
    while True:
        for x, _ in loader:
            yield x.to(device, non_blocking=True)


def local_batch(images: torch.Tensor, cfg: AffineConfig, eps: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = images.shape[0]
    device = images.device
    c = -0.6 + 1.2 * torch.rand(B, 2, device=device, dtype=images.dtype)
    # Put image at a random current control state by applying A then B.
    dirs0 = torch.tensor([[0, 1]], device=device, dtype=torch.long).repeat(B, 1)
    amounts0 = c.clone()
    x = apply_path_teacher(images, dirs0, amounts0, cfg)
    d = torch.randint(0, 2, (B,), device=device)
    target = torch.empty_like(x)
    for direction in (0, 1):
        mask = d == direction
        if mask.any():
            ep = torch.full((int(mask.sum()),), eps, device=device, dtype=images.dtype)
            xp = apply_affine_step(x[mask], direction, ep, cfg)
            xm = apply_affine_step(x[mask], direction, -ep, cfg)
            target[mask] = (xp - xm) / (2.0 * eps)
    return x, c, d, target


def sample_rollout(images: torch.Tensor, cfg: AffineConfig, low: float, high: float, family: str = "mixed", path_len: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = images.shape[0]
    device = images.device
    c0 = torch.zeros(B, 2, device=device, dtype=images.dtype)
    if family == "ab_ba":
        a = low + (high - low) * torch.rand(B, device=device, dtype=images.dtype)
        b = low + (high - low) * torch.rand(B, device=device, dtype=images.dtype)
        dab, mab, dba, mba = make_ab_ba(B, a, b, device)
        pick = torch.rand(B, device=device) < 0.5
        dirs = torch.where(pick[:, None], dab, dba)
        amounts = torch.where(pick[:, None], mab, mba)
    elif family == "loops":
        dlt = low + (high - low) * torch.rand(B, device=device, dtype=images.dtype)
        dirs, amounts = make_loop(B, dlt, device)
    elif family == "random":
        dirs, amounts = random_paths(B, path_len, low, high, device, signed=True)
    elif family == "mixed":
        # Use random paths as default; they include order information. For stability,
        # append enough zeros if path_len > generated family length.
        dirs, amounts = random_paths(B, path_len, low, high, device, signed=True)
    else:
        raise ValueError(family)
    y = apply_path_teacher(images, dirs, amounts, cfg)
    return images, c0, dirs, amounts, y


def sample_holonomy(images: torch.Tensor, cfg: AffineConfig, low: float, high: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = images.shape[0]
    device = images.device
    c0 = torch.zeros(B, 2, device=device, dtype=images.dtype)
    delta = low + (high - low) * torch.rand(B, device=device, dtype=images.dtype)
    dirs, amounts = make_loop(B, delta, device)
    y = apply_path_teacher(images, dirs, amounts, cfg)
    return images, c0, dirs, amounts, y - images


def image_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    af = a.flatten(1)
    bf = b.flatten(1)
    return ((af * bf).sum(-1) / (af.norm(dim=-1) * bf.norm(dim=-1) + eps)).mean()


@torch.no_grad()
def eval_model(model: ConvVF, image_iter: Iterator[torch.Tensor], cfg: AffineConfig, device: torch.device, batches: int, batch_size: int, low: float, high: float, model_steps: int) -> dict[str, float]:
    model.eval()
    acc: dict[str, float] = {}
    for _ in range(batches):
        img = next(image_iter)[:batch_size].to(device)
        B = img.shape[0]
        c0 = torch.zeros(B, 2, device=device, dtype=img.dtype)
        a = low + (high - low) * torch.rand(B, device=device, dtype=img.dtype)
        b = low + (high - low) * torch.rand(B, device=device, dtype=img.dtype)
        dab, mab, dba, mba = make_ab_ba(B, a, b, device)
        gt_ab = apply_path_teacher(img, dab, mab, cfg)
        gt_ba = apply_path_teacher(img, dba, mba, cfg)
        pr_ab, _ = integrate_path_model(model, img, c0, dab, mab, n_steps=model_steps)
        pr_ba, _ = integrate_path_model(model, img, c0, dba, mba, n_steps=model_steps)
        tc = gt_ab - gt_ba
        pc = pr_ab - pr_ba
        dirs, amounts = make_loop(B, low + (high - low) * torch.rand(B, device=device, dtype=img.dtype), device)
        gt_loop = apply_path_teacher(img, dirs, amounts, cfg)
        pr_loop, _ = integrate_path_model(model, img, c0, dirs, amounts, n_steps=model_steps)
        tr = gt_loop - img
        rr = pr_loop - img
        vals = {
            "ab_ba_endpoint_mse": float(0.5 * (F.mse_loss(pr_ab, gt_ab) + F.mse_loss(pr_ba, gt_ba)).cpu()),
            "ab_ba_comm_mse": float(F.mse_loss(pc, tc).cpu()),
            "ab_ba_comm_true_norm": float(tc.flatten(1).norm(dim=-1).mean().cpu()),
            "ab_ba_comm_pred_norm": float(pc.flatten(1).norm(dim=-1).mean().cpu()),
            "ab_ba_comm_cosine": float(image_cosine(pc, tc).cpu()),
            "hol_mse": float(F.mse_loss(rr, tr).cpu()),
            "hol_true_norm": float(tr.flatten(1).norm(dim=-1).mean().cpu()),
            "hol_pred_norm": float(rr.flatten(1).norm(dim=-1).mean().cpu()),
            "hol_cosine": float(image_cosine(rr, tr).cpu()),
            "false_identity_mse": float(rr.square().flatten(1).sum(-1).mean().cpu()),
        }
        for k, v in vals.items():
            acc[k] = acc.get(k, 0.0) + v
    return {k: v / batches for k, v in acc.items()}


def save_visual_grid(model: ConvVF, image_iter: Iterator[torch.Tensor], cfg: AffineConfig, out: Path, device: torch.device, model_steps: int) -> None:
    if vutils is None:
        return
    model.eval()
    img = next(image_iter)[:8].to(device)
    B = img.shape[0]
    c0 = torch.zeros(B, 2, device=device, dtype=img.dtype)
    a = torch.full((B,), 0.75, device=device, dtype=img.dtype)
    b = torch.full((B,), 0.75, device=device, dtype=img.dtype)
    dab, mab, dba, mba = make_ab_ba(B, a, b, device)
    gt_ab = apply_path_teacher(img, dab, mab, cfg)
    gt_ba = apply_path_teacher(img, dba, mba, cfg)
    with torch.no_grad():
        pr_ab, _ = integrate_path_model(model, img, c0, dab, mab, n_steps=model_steps)
        pr_ba, _ = integrate_path_model(model, img, c0, dba, mba, n_steps=model_steps)
    grid = torch.cat([img, gt_ab, pr_ab.clamp(0, 1), gt_ba, pr_ba.clamp(0, 1)], dim=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out, nrow=B)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU affine-MNIST benchmark for controlled holonomy connection models.")
    p.add_argument("--model", choices=["shared_cfm", "local_connection", "flat_pifm", "holonomy_connection"], default="holonomy_connection")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--outdir", default="runs/affine_mnist/holonomy_connection")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--train-low", type=float, default=0.2)
    p.add_argument("--train-high", type=float, default=0.8)
    p.add_argument("--ood-low", type=float, default=0.8)
    p.add_argument("--ood-high", type=float, default=1.2)
    p.add_argument("--model-steps", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=500, help="Run held-out paper metric suite every N training steps.")
    p.add_argument("--test-every", type=int, default=None, help="Alias for --eval-every; useful when thinking of held-out test metrics.")
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--local-weight", type=float, default=None)
    p.add_argument("--rollout-weight", type=float, default=None)
    p.add_argument("--holonomy-weight", type=float, default=None)
    p.add_argument("--cycle-weight", type=float, default=None)
    p.add_argument("--flat-weight", type=float, default=None)
    p.add_argument("--rot-deg-per-unit", type=float, default=35.0)
    p.add_argument("--translate-px-per-unit", type=float, default=7.0)
    p.add_argument("--save-best-checkpoints", action=argparse.BooleanOptionalAction, default=True, help="Save best_* checkpoints for key held-out paper metrics.")
    add_wandb_args(p)
    args = p.parse_args()
    if args.test_every is not None:
        args.eval_every = args.test_every
    return args


def default_weights(model: str) -> dict[str, float]:
    if model == "shared_cfm":
        return {"local": 1.0, "rollout": 0.0, "holonomy": 0.0, "cycle": 0.0, "flat": 0.0}
    if model == "local_connection":
        return {"local": 1.0, "rollout": 1.0, "holonomy": 0.0, "cycle": 0.02, "flat": 0.0}
    if model == "flat_pifm":
        return {"local": 1.0, "rollout": 0.0, "holonomy": 0.0, "cycle": 0.02, "flat": 0.5}
    if model == "holonomy_connection":
        return {"local": 1.0, "rollout": 1.0, "holonomy": 1.0, "cycle": 0.02, "flat": 0.0}
    raise ValueError(model)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    enable_fast_cuda(tf32=args.tf32, benchmark=True)
    if args.download_only:
        _require_torchvision()
        datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
        print(f"Downloaded MNIST to {args.data_dir}")
        return

    device = get_device(args.device)
    cfg = AffineConfig(rot_deg_per_unit=args.rot_deg_per_unit, translate_px_per_unit=args.translate_px_per_unit)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_loader = mnist_loader(args.data_dir, args.batch_size, args.workers, train=True, download=args.download)
    test_loader = mnist_loader(args.data_dir, args.eval_batch_size, args.workers, train=False, download=args.download)
    train_iter = infinite_images(train_loader, device)
    test_iter = infinite_images(test_loader, device)

    model = ConvVF(hidden=args.hidden, depth=args.depth).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, enabled=args.amp, dtype=args.amp_dtype)
    weights = default_weights(args.model)
    for k in ["local", "rollout", "holonomy", "cycle", "flat"]:
        override = getattr(args, f"{k}_weight")
        if override is not None:
            weights[k] = override

    run_config = {"args": vars(args), "weights": weights, "affine_cfg": cfg.__dict__, "num_parameters": count_parameters(model)}
    save_json(run_config, outdir / "config.json")
    wandb_run = init_wandb(args, config=run_config, device=device, run_kind="affine_mnist_train", model=model)
    if wandb_run is not None:
        wandb_run.define_metric("train/step")
        wandb_run.define_metric("train/*", step_metric="train/step")
        wandb_run.define_metric("test/*", step_metric="train/step")
        wandb_run.define_metric("paper/*", step_metric="train/step")
        wandb_run.define_metric("best/*", step_metric="train/step")

    best_tracker = BestTracker()
    metrics_path = outdir / "metrics.csv"
    pbar = trange(1, args.steps + 1, desc=f"affine_mnist/{args.model}", mininterval=1.0)
    last: dict[str, float] = {}
    for step in pbar:
        model.train()
        img = next(train_iter)
        terms: dict[str, torch.Tensor] = {}
        with autocast_context(device, enabled=args.amp, dtype=args.amp_dtype):
            if weights["local"]:
                x, c, d, target = local_batch(img, cfg)
                pred = torch.empty_like(target)
                for direction in (0, 1):
                    mask = d == direction
                    if mask.any():
                        pred[mask] = model.velocity(x[mask], c[mask], direction).to(dtype=pred.dtype)
                terms["local_loss"] = F.mse_loss(pred, target)
            if weights["rollout"]:
                x0, c0, dirs, amounts, y = sample_rollout(img, cfg, args.train_low, args.train_high, family="mixed")
                pred, _ = integrate_path_model(model, x0, c0, dirs, amounts, n_steps=args.model_steps)
                terms["rollout_loss"] = F.mse_loss(pred, y)
            if weights["holonomy"]:
                x0, c0, dirs, amounts, residual = sample_holonomy(img, cfg, args.train_low, args.train_high)
                pred, _ = integrate_path_model(model, x0, c0, dirs, amounts, n_steps=args.model_steps)
                terms["holonomy_loss"] = F.mse_loss(pred - x0, residual)
            if weights["cycle"]:
                B = img.shape[0]
                c0 = torch.zeros(B, 2, device=device, dtype=img.dtype)
                delta = args.train_low + (args.train_high - args.train_low) * torch.rand(B, device=device, dtype=img.dtype)
                direction = torch.randint(0, 2, (B,), device=device)
                dirs = torch.stack([direction, direction], dim=1)
                amounts = torch.stack([delta, -delta], dim=1)
                pred, _ = integrate_path_model(model, img, c0, dirs, amounts, n_steps=args.model_steps)
                terms["cycle_loss"] = F.mse_loss(pred, img)
            if weights["flat"]:
                B = img.shape[0]
                c0 = torch.zeros(B, 2, device=device, dtype=img.dtype)
                a = args.train_low + (args.train_high - args.train_low) * torch.rand(B, device=device, dtype=img.dtype)
                b = args.train_low + (args.train_high - args.train_low) * torch.rand(B, device=device, dtype=img.dtype)
                dab, mab, dba, mba = make_ab_ba(B, a, b, device)
                pr_ab, _ = integrate_path_model(model, img, c0, dab, mab, n_steps=args.model_steps)
                pr_ba, _ = integrate_path_model(model, img, c0, dba, mba, n_steps=args.model_steps)
                terms["flat_loss"] = F.mse_loss(pr_ab, pr_ba)
            loss = sum(weights[name.replace("_loss", "")] * val for name, val in terms.items())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 25 == 0:
            last = {k: float(v.detach().cpu()) for k, v in terms.items()}
            last["loss"] = float(loss.detach().cpu())
            last["grad_norm"] = float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm)
            last["lr"] = float(opt.param_groups[0]["lr"])
            pbar.set_postfix(loss=f"{last['loss']:.3e}", hol=f"{last.get('holonomy_loss',0):.2e}")
            if wandb_run is not None:
                wandb_run.log({"train/step": step, **{f"train/{k}": v for k, v in last.items()}}, step=step)
        if step == args.steps or (args.eval_every and step % args.eval_every == 0):
            vals = eval_model(model, test_iter, cfg, device, args.eval_batches, args.eval_batch_size, args.train_low, args.train_high, args.model_steps)
            ood = eval_model(model, test_iter, cfg, device, args.eval_batches, args.eval_batch_size, args.ood_low, args.ood_high, args.model_steps)
            merged_vals = {**vals, **{"ood_" + k: v for k, v in ood.items()}}
            paper_metrics = paper_metric_aliases(merged_vals)
            best_source = dict(paper_metrics)
            # Affine-MNIST is intentionally noncommuting, so identity residual is a
            # diagnostic curve, not a model-selection target.
            best_source.pop("paper/loop_pred_identity_mse", None)
            best_source.pop("paper/ood_loop_pred_identity_mse", None)
            best_logs, improved = best_tracker.update(best_source, step)
            row: dict[str, Any] = {"step": step, **last, **merged_vals, **{k.replace("/", "_"): v for k, v in paper_metrics.items()}, **{k.replace("/", "_"): v for k, v in best_logs.items()}}
            append_csv(row, metrics_path)
            checkpoint = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "affine_cfg": cfg.__dict__,
                "weights": weights,
                "step": step,
                "evals": merged_vals,
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
                for key in improved:
                    if key in important:
                        atomic_torch_save(checkpoint, outdir / important[key])
            viz_path = outdir / f"viz_step_{step}.png"
            save_visual_grid(model, test_iter, cfg, viz_path, device, args.model_steps)
            if wandb_run is not None:
                log_dict = {
                    "train/step": step,
                    **{f"test/{k}": float(v) for k, v in merged_vals.items()},
                    **paper_metrics,
                    **best_logs,
                }
                try:
                    import wandb  # type: ignore
                    if viz_path.exists():
                        log_dict["viz/ab_ba_grid"] = wandb.Image(str(viz_path))
                except Exception:
                    pass
                wandb_run.log(log_dict, step=step)
    if wandb_run is not None:
        if args.wandb_log_artifacts:
            wandb_log_artifacts(wandb_run, outdir, name=f"affine-mnist-{args.model}-seed{args.seed}", aliases=["latest", f"step-{args.steps}"])
        wandb_run.finish()
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
