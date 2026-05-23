from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

ModelName = Literal["independent_cfm", "shared_cfm", "local_connection", "flat_pifm", "holonomy_connection"]


def activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(name)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 4, act: str = "silu"):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(d, hidden), activation(act)])
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class VFConfig:
    model: ModelName = "holonomy_connection"
    hidden: int = 128
    depth: int = 4
    act: str = "silu"


class ControlledVectorField(nn.Module):
    """Base interface."""
    model_name: str

    def velocity(self, x: torch.Tensor, controls: torch.Tensor, direction: int) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, controls: torch.Tensor, direction: int) -> torch.Tensor:
        return self.velocity(x, controls, direction)


class IndependentCFM(ControlledVectorField):
    """Two independent vector fields V_A(x), V_B(x). No control coordinates.

    This is a serious baseline: if the world is just rotation+translation, this
    can compose A and B perfectly. It should fail on control-dependent connections.
    """
    model_name = "independent_cfm"

    def __init__(self, cfg: VFConfig):
        super().__init__()
        self.A = MLP(2, 2, cfg.hidden, cfg.depth, cfg.act)
        self.B = MLP(2, 2, cfg.hidden, cfg.depth, cfg.act)

    def velocity(self, x: torch.Tensor, controls: torch.Tensor, direction: int) -> torch.Tensor:
        return self.A(x) if direction == 0 else self.B(x)


class SharedCFM(ControlledVectorField):
    """One shared vector field V(x, alpha, beta, direction_token)."""
    model_name = "shared_cfm"

    def __init__(self, cfg: VFConfig):
        super().__init__()
        self.mlp = MLP(2 + 2 + 2, 2, cfg.hidden, cfg.depth, cfg.act)

    def velocity(self, x: torch.Tensor, controls: torch.Tensor, direction: int) -> torch.Tensor:
        B = x.shape[0]
        tok = torch.zeros(B, 2, device=x.device, dtype=x.dtype)
        tok[:, direction] = 1.0
        return self.mlp(torch.cat([x, controls, tok], dim=-1))


# The architecture is intentionally the same for shared/local/flat/holonomy.
# The difference is training objective, not raw capacity.
LocalConnection = SharedCFM
FlatPiFM = SharedCFM
HolonomyConnection = SharedCFM


def build_model(cfg: VFConfig) -> ControlledVectorField:
    if cfg.model == "independent_cfm":
        return IndependentCFM(cfg)
    if cfg.model in {"shared_cfm", "local_connection", "flat_pifm", "holonomy_connection"}:
        m = SharedCFM(cfg)
        m.model_name = cfg.model  # type: ignore[attr-defined]
        return m
    raise ValueError(cfg.model)
