import torch

from holoflow_conn.integrators import integrate_path, make_loop
from holoflow_conn.worlds import WorldConfig, gt_velocity, sample_base


def test_commuting_loop_is_zero():
    torch.manual_seed(0)
    device = torch.device('cpu')
    cfg = WorldConfig(world='commute')
    x = sample_base(128, device)
    c = torch.zeros(128, 2)
    d = torch.full((128,), 0.5)
    dirs, amounts = make_loop(128, d, device)
    y, _ = integrate_path(lambda xx, cc, dd: gt_velocity(xx, cc, dd, cfg), x, c, dirs, amounts, n_steps=8)
    assert torch.mean((y - x).norm(dim=-1)).item() < 1e-5


def test_nonlinear_loop_nonzero():
    torch.manual_seed(0)
    device = torch.device('cpu')
    cfg = WorldConfig(world='nonlinear')
    x = sample_base(128, device)
    c = torch.zeros(128, 2)
    d = torch.full((128,), 0.5)
    dirs, amounts = make_loop(128, d, device)
    y, _ = integrate_path(lambda xx, cc, dd: gt_velocity(xx, cc, dd, cfg), x, c, dirs, amounts, n_steps=16)
    assert torch.mean((y - x).norm(dim=-1)).item() > 1e-3
