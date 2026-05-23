from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .data import sample_base, sample_local_batch, sample_rollout_batch, sample_holonomy_batch
from .integrators import integrate_path, make_ab_ba, make_loop
from .models import ControlledVectorField
from .worlds import WorldConfig, gt_velocity
from .losses import local_velocity_loss, flatness_penalty


def cosine_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return ((a * b).sum(dim=-1) / denom).mean()


@torch.no_grad()
def _endpoint_errors(
    model: ControlledVectorField,
    cfg: WorldConfig,
    device: torch.device,
    batch: int,
    low: float,
    high: float,
    n_steps_model: int,
    n_steps_gt: int,
    kind: str,
) -> dict[str, float]:
    x0 = sample_base(batch, device)
    c0 = torch.zeros(batch, 2, device=device)
    if kind == "ab_ba":
        a = low + (high - low) * torch.rand(batch, device=device)
        b = low + (high - low) * torch.rand(batch, device=device)
        dirs_ab, amt_ab, dirs_ba, amt_ba = make_ab_ba(batch, a, b, device)
        gt_ab, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dirs_ab, amt_ab, n_steps=n_steps_gt)
        gt_ba, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dirs_ba, amt_ba, n_steps=n_steps_gt)
        pr_ab, _ = integrate_path(model.velocity, x0, c0, dirs_ab, amt_ab, n_steps=n_steps_model)
        pr_ba, _ = integrate_path(model.velocity, x0, c0, dirs_ba, amt_ba, n_steps=n_steps_model)
        true_comm = gt_ab - gt_ba
        pred_comm = pr_ab - pr_ba
        return {
            f"{kind}_endpoint_mse": float(0.5 * (F.mse_loss(pr_ab, gt_ab) + F.mse_loss(pr_ba, gt_ba)).cpu()),
            f"{kind}_comm_mse": float(F.mse_loss(pred_comm, true_comm).cpu()),
            f"{kind}_comm_true_norm": float(true_comm.norm(dim=-1).mean().cpu()),
            f"{kind}_comm_pred_norm": float(pred_comm.norm(dim=-1).mean().cpu()),
            f"{kind}_comm_cosine": float(cosine_mean(pred_comm, true_comm).cpu()),
        }
    if kind == "random":
        rb = sample_rollout_batch(batch, cfg, device, family="random", amount_low=low, amount_high=high, signed=True, path_len=4, gt_steps=n_steps_gt)
        pred, _ = integrate_path(model.velocity, rb.x0, rb.c0, rb.dirs, rb.amounts, n_steps=n_steps_model)
        return {f"{kind}_endpoint_mse": float(F.mse_loss(pred, rb.y).cpu())}
    raise ValueError(kind)


@torch.no_grad()
def _holonomy_metrics(model: ControlledVectorField, cfg: WorldConfig, device: torch.device, batch: int, low: float, high: float, n_steps_model: int, n_steps_gt: int, prefix: str) -> dict[str, float]:
    hb = sample_holonomy_batch(batch, cfg, device, delta_low=low, delta_high=high, gt_steps=n_steps_gt)
    pred, _ = integrate_path(model.velocity, hb.x0, hb.c0, hb.dirs, hb.amounts, n_steps=n_steps_model)
    pred_res = pred - hb.x0
    true_res = hb.true_residual
    return {
        f"{prefix}_hol_mse": float(F.mse_loss(pred_res, true_res).cpu()),
        f"{prefix}_hol_true_norm": float(true_res.norm(dim=-1).mean().cpu()),
        f"{prefix}_hol_pred_norm": float(pred_res.norm(dim=-1).mean().cpu()),
        f"{prefix}_hol_cosine": float(cosine_mean(pred_res, true_res).cpu()),
        f"{prefix}_false_identity_mse": float(pred_res.square().sum(dim=-1).mean().cpu()),
    }


def evaluate_model(
    model: ControlledVectorField,
    cfg: WorldConfig,
    device: torch.device,
    batch_size: int = 2048,
    batches: int = 4,
    train_low: float = 0.15,
    train_high: float = 0.75,
    ood_low: float = 0.75,
    ood_high: float = 1.15,
    n_steps_model: int = 8,
    n_steps_gt: int = 24,
    flat_batch: int = 128,
) -> dict[str, float]:
    model.eval()
    acc: dict[str, float] = {}
    for _ in range(batches):
        # local MSE with gradient disabled except flatness diagnostic below.
        lb = sample_local_batch(batch_size, cfg, device)
        with torch.no_grad():
            loc = local_velocity_loss(model, lb)
        vals: dict[str, float] = {"local_mse": float(loc.cpu())}
        vals.update(_endpoint_errors(model, cfg, device, batch_size, train_low, train_high, n_steps_model, n_steps_gt, "ab_ba"))
        vals.update(_endpoint_errors(model, cfg, device, batch_size, train_low, train_high, n_steps_model, n_steps_gt, "random"))
        vals.update(_holonomy_metrics(model, cfg, device, batch_size, train_low, train_high, n_steps_model, n_steps_gt, "id"))
        ood_ab = _endpoint_errors(model, cfg, device, batch_size, ood_low, ood_high, n_steps_model, n_steps_gt, "ab_ba")
        vals.update({"ood_" + k: v for k, v in ood_ab.items()})
        ood_hol = _holonomy_metrics(model, cfg, device, batch_size, ood_low, ood_high, n_steps_model, n_steps_gt, "ood")
        vals.update(ood_hol)
        # Flatness diagnostic; autograd required.
        try:
            x_flat = sample_base(flat_batch, device)
            c_flat = -0.6 + 1.2 * torch.rand(flat_batch, 2, device=device)
            flat = flatness_penalty(model, x_flat, c_flat)
            vals["flatness_norm2"] = float(flat.detach().cpu())
        except Exception:
            vals["flatness_norm2"] = float("nan")

        for k, v in vals.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    return {k: v / batches for k, v in acc.items()}


def holonomy_scaling(
    model: ControlledVectorField,
    cfg: WorldConfig,
    device: torch.device,
    deltas: list[float],
    batch: int = 2048,
    n_steps_model: int = 8,
    n_steps_gt: int = 32,
) -> list[dict[str, float]]:
    model.eval()
    rows = []
    for d in deltas:
        hb = sample_holonomy_batch(batch, cfg, device, delta_low=d, delta_high=d, gt_steps=n_steps_gt)
        with torch.no_grad():
            pred, _ = integrate_path(model.velocity, hb.x0, hb.c0, hb.dirs, hb.amounts, n_steps=n_steps_model)
        pr = pred - hb.x0
        tr = hb.true_residual
        rows.append(
            {
                "delta": d,
                "hol_mse": float(F.mse_loss(pr, tr).cpu()),
                "true_norm": float(tr.norm(dim=-1).mean().cpu()),
                "pred_norm": float(pr.norm(dim=-1).mean().cpu()),
                "cosine": float(cosine_mean(pr, tr).cpu()),
            }
        )
    return rows
