import torch

from holoflow_conn.models import VFConfig, build_model
from holoflow_conn.integrators import integrate_direction


def test_model_velocity_and_integrate_shapes():
    torch.set_num_threads(1)
    model = build_model(VFConfig(model='holonomy_connection', hidden=16, depth=2))
    x = torch.randn(8, 2)
    c = torch.zeros(8, 2)
    v = model.velocity(x, c, 0)
    assert v.shape == (8, 2)
    y, cc = integrate_direction(model.velocity, x, c, 0, torch.ones(8) * 0.2, n_steps=2)
    assert y.shape == (8, 2)
    assert cc.shape == (8, 2)
