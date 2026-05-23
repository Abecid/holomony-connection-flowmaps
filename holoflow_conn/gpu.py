from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def enable_fast_cuda(tf32: bool = True, benchmark: bool = True) -> None:
    """Enable safe CUDA speed knobs for A100/A800-class GPUs.

    TF32 is usually the right default for this project: losses are regression
    losses with broad tolerances, and TF32 gives large matmul speedups on Ampere.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(benchmark)
    try:
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        pass


def autocast_context(device: torch.device, enabled: bool, dtype: str = "bf16") -> Any:
    if not enabled or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype in {"bf16", "bfloat16"} else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool, dtype: str = "bf16") -> torch.amp.GradScaler:
    # BF16 does not need gradient scaling. FP16 does.
    use_scaler = enabled and device.type == "cuda" and dtype in {"fp16", "float16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except TypeError:  # older torch fallback
        return torch.cuda.amp.GradScaler(enabled=use_scaler)
