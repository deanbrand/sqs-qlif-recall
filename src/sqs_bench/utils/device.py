"""Device resolution and a short device description for the run record."""
from __future__ import annotations

import torch


def resolve_device(name: str | None = "auto") -> torch.device:
    """``"auto"`` -> cuda when available, else cpu. Anything else is passed through."""
    if name is None or str(name).lower() in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(str(name))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
    return dev


def describe_device(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return str(device)
