from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import LocalBatch
from .integrators import integrate_path
from .models import ControlledVectorField


@dataclass
class LossWeights:
    local: float = 1.0
    rollout: float = 0.0
    holonomy: float = 0.0
    cycle: float = 0.0
    flat: float = 0.0


def default_weights(model_name: str) -> LossWeights:
    if model_name == "independent_cfm":
        return LossWeights(local=1.0)
    if model_name == "shared_cfm":
        return LossWeights(local=1.0)
    if model_name == "local_connection":
        return LossWeights(local=1.0, rollout=1.0, cycle=0.05)
    if model_name == "flat_pifm":
        # The point of this baseline is the path-independence / zero-curvature bias.
        return LossWeights(local=1.0, cycle=0.05, flat=0.25)
    if model_name == "holonomy_connection":
        return LossWeights(local=1.0, rollout=1.0, holonomy=1.0, cycle=0.05)
    raise ValueError(model_name)


def local_velocity_loss(model: ControlledVectorField, batch: LocalBatch) -> torch.Tensor:
    pred = torch.empty_like(batch.target_v)
    for direction in (0, 1):
        mask = batch.direction == direction
        if mask.any():
            pred[mask] = model.velocity(batch.x[mask], batch.controls[mask], direction).to(dtype=pred.dtype)
    return F.mse_loss(pred, batch.target_v)


def rollout_loss(model: ControlledVectorField, x0: torch.Tensor, c0: torch.Tensor, dirs: torch.Tensor, amounts: torch.Tensor, y: torch.Tensor, n_steps: int) -> torch.Tensor:
    pred, _ = integrate_path(model.velocity, x0, c0, dirs, amounts, n_steps=n_steps, method="rk4")
    return F.mse_loss(pred, y)


def holonomy_loss(model: ControlledVectorField, x0: torch.Tensor, c0: torch.Tensor, dirs: torch.Tensor, amounts: torch.Tensor, true_residual: torch.Tensor, n_steps: int) -> torch.Tensor:
    pred, _ = integrate_path(model.velocity, x0, c0, dirs, amounts, n_steps=n_steps, method="rk4")
    return F.mse_loss(pred - x0, true_residual)


def cycle_loss(model: ControlledVectorField, x0: torch.Tensor, c0: torch.Tensor, dirs: torch.Tensor, amounts: torch.Tensor, n_steps: int) -> torch.Tensor:
    pred, _ = integrate_path(model.velocity, x0, c0, dirs, amounts, n_steps=n_steps, method="rk4")
    return F.mse_loss(pred, x0)


def flatness_penalty(model: ControlledVectorField, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Differentiable Lie-bracket/curvature penalty.

    Omega_AB = partial_alpha V_B - partial_beta V_A + D_x V_B[V_A] - D_x V_A[V_B].

    This is intentionally small-batch; it is only a baseline regularizer and a
    diagnostic, not the main training bottleneck.
    """
    x = x.detach().requires_grad_(True)
    c = c.detach().requires_grad_(True)
    VA = model.velocity(x, c, 0)
    VB = model.velocity(x, c, 1)
    comps = []
    for k in range(2):
        grad_VB_c = torch.autograd.grad(VB[:, k].sum(), c, create_graph=True, retain_graph=True)[0]
        grad_VA_c = torch.autograd.grad(VA[:, k].sum(), c, create_graph=True, retain_graph=True)[0]
        grad_VB_x = torch.autograd.grad(VB[:, k].sum(), x, create_graph=True, retain_graph=True)[0]
        grad_VA_x = torch.autograd.grad(VA[:, k].sum(), x, create_graph=True, retain_graph=True)[0]
        partial_alpha_VB = grad_VB_c[:, 0]
        partial_beta_VA = grad_VA_c[:, 1]
        jvb_va = (grad_VB_x * VA).sum(dim=-1)
        jva_vb = (grad_VA_x * VB).sum(dim=-1)
        comps.append(partial_alpha_VB - partial_beta_VA + jvb_va - jva_vb)
    omega = torch.stack(comps, dim=-1)
    return (omega.square().sum(dim=-1)).mean()
