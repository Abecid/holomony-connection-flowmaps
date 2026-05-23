"""Ground-truth controlled vector fields for holonomy experiments.

The central object is a controlled connection over a two-dimensional control
space c=(alpha,beta). Direction 0 integrates in alpha; direction 1 integrates
in beta. Nonzero commutator/curvature means the endpoint depends on the order
of A/B controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

WorldName = Literal["se2", "body", "nonlinear", "commute"]


@dataclass(frozen=True)
class WorldConfig:
    world: WorldName = "nonlinear"
    rot_scale: float = 1.0
    trans_scale: float = 0.75
    shear_scale: float = 0.35
    nonlinear_scale: float = 0.25


def J(x: torch.Tensor) -> torch.Tensor:
    """90-degree rotation generator J x = (-y, x)."""
    return torch.stack([-x[:, 1], x[:, 0]], dim=-1)


def sample_base(batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Asymmetric 2D Gaussian mixture. Asymmetry prevents distribution metrics
    from hiding order effects.
    """
    centers = torch.tensor(
        [[-1.25, -0.55], [0.95, -0.9], [-0.3, 1.15], [1.35, 0.8], [-1.45, 1.25]],
        device=device,
        dtype=dtype,
    )
    probs = torch.tensor([0.20, 0.25, 0.18, 0.25, 0.12], device=device, dtype=dtype)
    idx = torch.multinomial(probs, batch_size, replacement=True)
    return centers[idx] + 0.11 * torch.randn(batch_size, 2, device=device, dtype=dtype)


def gt_velocity(x: torch.Tensor, controls: torch.Tensor, direction: int | torch.Tensor, cfg: WorldConfig) -> torch.Tensor:
    """Batched ground-truth velocity.

    Args:
        x: [B,2] state.
        controls: [B,2] current (alpha,beta).
        direction: int or [B] long, 0=A/alpha, 1=B/beta.
        cfg: world configuration.
    """
    a = controls[:, 0]
    b = controls[:, 1]
    device, dtype = x.device, x.dtype

    if cfg.world == "commute":
        v_a = torch.stack([torch.ones_like(a) * cfg.trans_scale, torch.zeros_like(a)], dim=-1)
        v_b = torch.stack([torch.zeros_like(a), torch.ones_like(a) * cfg.trans_scale], dim=-1)
    elif cfg.world == "se2":
        v_a = cfg.rot_scale * J(x)
        v_b = torch.stack([torch.ones_like(a) * cfg.trans_scale, torch.ones_like(a) * 0.25 * cfg.trans_scale], dim=-1)
    elif cfg.world == "body":
        # Body-frame translation direction rotates with alpha; x-dependence enters through A.
        v_a = cfg.rot_scale * J(x)
        angle = cfg.rot_scale * a
        v_b = cfg.trans_scale * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    elif cfg.world == "nonlinear":
        # A nonlinear, control-dependent non-Abelian connection.
        # Independent CFM cannot represent this because both directions depend on controls.
        rot_rate = cfg.rot_scale * (1.0 + 0.25 * torch.sin(1.7 * b))
        swirl = rot_rate.unsqueeze(-1) * J(x)
        a_bend = cfg.nonlinear_scale * torch.stack(
            [torch.sin(0.8 * x[:, 1] + a + 0.5 * b), torch.cos(0.7 * x[:, 0] - 0.3 * a + b)],
            dim=-1,
        )
        v_a = swirl + a_bend

        # B: translation direction depends on alpha, plus a small shear/nonlinear term.
        angle = 1.25 * a + 0.25 * torch.sin(b)
        trans = cfg.trans_scale * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        shear = cfg.shear_scale * (1.0 + 0.2 * torch.cos(a - b)).unsqueeze(-1) * torch.stack(
            [x[:, 1], torch.zeros_like(a)], dim=-1
        )
        b_bend = cfg.nonlinear_scale * torch.stack(
            [0.7 * torch.sin(x[:, 0] + b), -0.5 * torch.cos(x[:, 1] - a)], dim=-1
        )
        v_b = trans + shear + b_bend
    else:  # pragma: no cover
        raise ValueError(f"unknown world: {cfg.world}")

    if isinstance(direction, int):
        return v_a if direction == 0 else v_b
    direction = direction.to(device=device)
    return torch.where((direction == 0).unsqueeze(-1), v_a, v_b)


def direction_name(direction: int) -> str:
    return "A" if direction == 0 else "B"
