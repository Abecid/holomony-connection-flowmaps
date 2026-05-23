from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import sample_base, sample_holonomy_batch
from .integrators import integrate_path, make_ab_ba
from .models import ControlledVectorField
from .worlds import WorldConfig, gt_velocity


@torch.no_grad()
def plot_ab_ba_clouds(model: ControlledVectorField, cfg: WorldConfig, path: str | Path, device: torch.device, n: int = 1024, amount: float = 0.7, model_steps: int = 8, gt_steps: int = 32) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x0 = sample_base(n, device)
    c0 = torch.zeros(n, 2, device=device)
    a = torch.full((n,), amount, device=device)
    b = torch.full((n,), amount, device=device)
    dab, mab, dba, mba = make_ab_ba(n, a, b, device)
    gt_ab, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dab, mab, n_steps=gt_steps)
    gt_ba, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dba, mba, n_steps=gt_steps)
    pr_ab, _ = integrate_path(model.velocity, x0, c0, dab, mab, n_steps=model_steps)
    pr_ba, _ = integrate_path(model.velocity, x0, c0, dba, mba, n_steps=model_steps)
    xs = [x0, gt_ab, gt_ba, pr_ab, pr_ba]
    titles = ["source", "true AB", "true BA", "pred AB", "pred BA"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), constrained_layout=True)
    for ax, pts, title in zip(axes, xs, titles):
        pts = pts.cpu()
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_holonomy_arrows(model: ControlledVectorField, cfg: WorldConfig, path: str | Path, device: torch.device, amount: float = 0.45, model_steps: int = 8, gt_steps: int = 32) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = torch.linspace(-2.0, 2.0, 17, device=device)
    ys = torch.linspace(-2.0, 2.0, 17, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    x0 = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    c0 = torch.zeros(x0.shape[0], 2, device=device)
    delta = torch.full((x0.shape[0],), amount, device=device)
    hb = sample_holonomy_batch(x0.shape[0], cfg, device, delta_low=amount, delta_high=amount, gt_steps=gt_steps)
    # Override sampled points to grid.
    dirs, amounts = hb.dirs, hb.amounts
    gt, _ = integrate_path(lambda x, c, d: gt_velocity(x, c, d, cfg), x0, c0, dirs, amounts, n_steps=gt_steps)
    pr, _ = integrate_path(model.velocity, x0, c0, dirs, amounts, n_steps=model_steps)
    gt_res = (gt - x0).cpu()
    pr_res = (pr - x0).cpu()
    xcpu = x0.cpu()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, res, title in [(axes[0], gt_res, "true loop residual"), (axes[1], pr_res, "predicted loop residual")]:
        ax.quiver(xcpu[:, 0], xcpu[:, 1], res[:, 0], res[:, 1], angles="xy", scale_units="xy", scale=1.0, width=0.003)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
