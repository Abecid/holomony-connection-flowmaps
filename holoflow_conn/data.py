from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .integrators import integrate_path, make_ab_ba, make_loop
from .worlds import WorldConfig, gt_velocity, sample_base

PathFamily = Literal["ab_ba", "loops", "random", "mixed"]


@dataclass
class LocalBatch:
    x: torch.Tensor
    controls: torch.Tensor
    direction: torch.Tensor
    target_v: torch.Tensor


@dataclass
class RolloutBatch:
    x0: torch.Tensor
    c0: torch.Tensor
    dirs: torch.Tensor
    amounts: torch.Tensor
    y: torch.Tensor


@dataclass
class HolonomyBatch:
    x0: torch.Tensor
    c0: torch.Tensor
    delta: torch.Tensor
    dirs: torch.Tensor
    amounts: torch.Tensor
    true_residual: torch.Tensor
    true_endpoint: torch.Tensor


def sample_controls(batch: int, device: torch.device, low: float = -0.6, high: float = 0.6) -> torch.Tensor:
    return low + (high - low) * torch.rand(batch, 2, device=device)


def sample_local_batch(
    batch: int,
    cfg: WorldConfig,
    device: torch.device,
    control_low: float = -0.6,
    control_high: float = 0.6,
) -> LocalBatch:
    x = sample_base(batch, device)
    c = sample_controls(batch, device, control_low, control_high)
    d = torch.randint(0, 2, (batch,), device=device)
    va = gt_velocity(x, c, 0, cfg)
    vb = gt_velocity(x, c, 1, cfg)
    target = torch.where((d == 0).unsqueeze(-1), va, vb)
    return LocalBatch(x=x, controls=c, direction=d, target_v=target)


def random_path(batch: int, length: int, device: torch.device, amount_low: float, amount_high: float, signed: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    dirs = torch.randint(0, 2, (batch, length), device=device)
    mags = amount_low + (amount_high - amount_low) * torch.rand(batch, length, device=device)
    if signed:
        signs = torch.where(torch.rand(batch, length, device=device) < 0.5, -1.0, 1.0)
        mags = mags * signs
    return dirs, mags


def sample_rollout_batch(
    batch: int,
    cfg: WorldConfig,
    device: torch.device,
    family: PathFamily = "mixed",
    amount_low: float = 0.15,
    amount_high: float = 0.75,
    signed: bool = True,
    path_len: int = 4,
    gt_steps: int = 16,
) -> RolloutBatch:
    x0 = sample_base(batch, device)
    # Starting at zero control isolates path-order effects around a common origin.
    c0 = torch.zeros(batch, 2, device=device)

    if family == "mixed":
        # half AB/BA, quarter loops, quarter random
        choices = torch.rand(batch, device=device)
        dirs = torch.zeros(batch, path_len, dtype=torch.long, device=device)
        amounts = torch.zeros(batch, path_len, device=device)
        # AB/BA subset
        mask_ab = choices < 0.5
        n_ab = int(mask_ab.sum().item())
        if n_ab:
            a = amount_low + (amount_high - amount_low) * torch.rand(n_ab, device=device)
            b = amount_low + (amount_high - amount_low) * torch.rand(n_ab, device=device)
            if signed:
                a = a * torch.where(torch.rand(n_ab, device=device) < 0.5, -1.0, 1.0)
                b = b * torch.where(torch.rand(n_ab, device=device) < 0.5, -1.0, 1.0)
            dab, mab, dba, mba = make_ab_ba(n_ab, a, b, device)
            pick = torch.rand(n_ab, device=device) < 0.5
            d2 = torch.where(pick[:, None], dab, dba)
            m2 = torch.where(pick[:, None], mab, mba)
            dirs[mask_ab, :2] = d2
            amounts[mask_ab, :2] = m2
        # loop subset
        mask_loop = (choices >= 0.5) & (choices < 0.75)
        n_loop = int(mask_loop.sum().item())
        if n_loop:
            delta = amount_low + (amount_high - amount_low) * torch.rand(n_loop, device=device)
            dl, ml = make_loop(n_loop, delta, device)
            dirs[mask_loop] = dl
            amounts[mask_loop] = ml
        # random subset
        mask_random = choices >= 0.75
        n_rand = int(mask_random.sum().item())
        if n_rand:
            dr, mr = random_path(n_rand, path_len, device, amount_low, amount_high, signed=signed)
            dirs[mask_random] = dr
            amounts[mask_random] = mr
    elif family == "ab_ba":
        a = amount_low + (amount_high - amount_low) * torch.rand(batch, device=device)
        b = amount_low + (amount_high - amount_low) * torch.rand(batch, device=device)
        if signed:
            a = a * torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
            b = b * torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
        dab, mab, dba, mba = make_ab_ba(batch, a, b, device)
        pick = torch.rand(batch, device=device) < 0.5
        dirs = torch.zeros(batch, path_len, dtype=torch.long, device=device)
        amounts = torch.zeros(batch, path_len, device=device)
        dirs[:, :2] = torch.where(pick[:, None], dab, dba)
        amounts[:, :2] = torch.where(pick[:, None], mab, mba)
    elif family == "loops":
        delta = amount_low + (amount_high - amount_low) * torch.rand(batch, device=device)
        dirs, amounts = make_loop(batch, delta, device)
    elif family == "random":
        dirs, amounts = random_path(batch, path_len, device, amount_low, amount_high, signed=signed)
    else:
        raise ValueError(family)

    y, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dirs, amounts, n_steps=gt_steps, method="rk4")
    return RolloutBatch(x0=x0, c0=c0, dirs=dirs, amounts=amounts, y=y)


def sample_holonomy_batch(
    batch: int,
    cfg: WorldConfig,
    device: torch.device,
    delta_low: float = 0.15,
    delta_high: float = 0.75,
    gt_steps: int = 16,
) -> HolonomyBatch:
    x0 = sample_base(batch, device)
    c0 = torch.zeros(batch, 2, device=device)
    delta = delta_low + (delta_high - delta_low) * torch.rand(batch, device=device)
    dirs, amounts = make_loop(batch, delta, device)
    y, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dirs, amounts, n_steps=gt_steps, method="rk4")
    return HolonomyBatch(x0=x0, c0=c0, delta=delta, dirs=dirs, amounts=amounts, true_residual=y - x0, true_endpoint=y)


def sample_cycle_batch(batch: int, device: torch.device, delta_low: float = 0.15, delta_high: float = 0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = sample_base(batch, device)
    c0 = torch.zeros(batch, 2, device=device)
    delta = delta_low + (delta_high - delta_low) * torch.rand(batch, device=device)
    direction = torch.randint(0, 2, (batch,), device=device)
    dirs = torch.stack([direction, direction], dim=1)
    amounts = torch.stack([delta, -delta], dim=1)
    return x0, c0, dirs, amounts
