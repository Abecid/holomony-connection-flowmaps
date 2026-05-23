from __future__ import annotations

from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def _rk4_step(vfn: VelocityFn, x: torch.Tensor, c: torch.Tensor, direction: int, h: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(h):
        h = torch.full((x.shape[0],), float(h), device=x.device, dtype=x.dtype)
    h_col = h.unsqueeze(-1)
    e = torch.zeros_like(c)
    e[:, direction] = 1.0
    k1 = vfn(x, c, direction)
    k2 = vfn(x + 0.5 * h_col * k1, c + 0.5 * h_col * e, direction)
    k3 = vfn(x + 0.5 * h_col * k2, c + 0.5 * h_col * e, direction)
    k4 = vfn(x + h_col * k3, c + h_col * e, direction)
    x_next = x + (h_col / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    c_next = c + h_col * e
    return x_next, c_next


def _euler_step(vfn: VelocityFn, x: torch.Tensor, c: torch.Tensor, direction: int, h: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(h):
        h = torch.full((x.shape[0],), float(h), device=x.device, dtype=x.dtype)
    h_col = h.unsqueeze(-1)
    e = torch.zeros_like(c)
    e[:, direction] = 1.0
    return x + h_col * vfn(x, c, direction), c + h_col * e


def integrate_direction(
    vfn: VelocityFn,
    x: torch.Tensor,
    controls: torch.Tensor,
    direction: int,
    amount: torch.Tensor | float,
    n_steps: int = 8,
    method: str = "rk4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a single signed control amount along direction A/B."""
    if not torch.is_tensor(amount):
        amount_t = torch.full((x.shape[0],), float(amount), device=x.device, dtype=x.dtype)
    else:
        amount_t = amount.to(device=x.device, dtype=x.dtype)
        if amount_t.ndim == 0:
            amount_t = amount_t.expand(x.shape[0])
    h = amount_t / float(n_steps)
    step = _rk4_step if method == "rk4" else _euler_step
    y, c = x, controls
    for _ in range(n_steps):
        y, c = step(vfn, y, c, direction, h)
    return y, c


def integrate_path(
    vfn: VelocityFn,
    x: torch.Tensor,
    controls: torch.Tensor,
    dirs: torch.Tensor,
    amounts: torch.Tensor,
    n_steps: int = 8,
    method: str = "rk4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a padded path. dirs/amounts are [B,L]."""
    y, c = x, controls
    L = dirs.shape[1]
    for j in range(L):
        # Split by direction to keep integrator direction an int.
        d = dirs[:, j]
        amt = amounts[:, j]
        mask_a = d == 0
        mask_b = d == 1
        y_next = y.clone()
        c_next = c.clone()
        if mask_a.any():
            ya, ca = integrate_direction(vfn, y[mask_a], c[mask_a], 0, amt[mask_a], n_steps=n_steps, method=method)
            y_next[mask_a], c_next[mask_a] = ya, ca
        if mask_b.any():
            yb, cb = integrate_direction(vfn, y[mask_b], c[mask_b], 1, amt[mask_b], n_steps=n_steps, method=method)
            y_next[mask_b], c_next[mask_b] = yb, cb
        y, c = y_next, c_next
    return y, c


def make_loop(batch: int, delta: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dirs = torch.tensor([[0, 1, 0, 1]], device=device, dtype=torch.long).repeat(batch, 1)
    amounts = torch.stack([delta, delta, -delta, -delta], dim=1)
    return dirs, amounts


def make_ab_ba(batch: int, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dirs_ab = torch.tensor([[0, 1]], device=device, dtype=torch.long).repeat(batch, 1)
    dirs_ba = torch.tensor([[1, 0]], device=device, dtype=torch.long).repeat(batch, 1)
    amt_ab = torch.stack([a, b], dim=1)
    amt_ba = torch.stack([b, a], dim=1)
    return dirs_ab, amt_ab, dirs_ba, amt_ba
