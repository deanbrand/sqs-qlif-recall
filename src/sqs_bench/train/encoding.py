"""Input encoding, applied on the fly to every batch on the compute device.

``identity`` is the only encoder: the memory probe is already a ``[B, T, D]``
spike train, so only the axes move. The implementation lives in
``sqs_bench.data.encoders``.
"""
from __future__ import annotations

from typing import Callable

from torch import Tensor

from sqs_bench.data import encoders

_ENCODERS: dict[str, Callable[..., Tensor]] = {"identity": encoders.identity}


def get_encoder(name: str) -> tuple[Callable[..., Tensor], str | None]:
    """``(callable, deviation_or_None)``; the deviation is always ``None`` here."""
    key = str(name)
    if key not in _ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; known: {sorted(_ENCODERS)}")
    return _ENCODERS[key], None


class BatchEncoder:
    """Turns a ``[B, T, D]`` batch into the ``[T, B, D]`` train the network consumes."""

    def __init__(self, name: str, T: int) -> None:
        self.name = str(name)
        self.T = int(T)
        self.fn, dev = get_encoder(self.name)
        self.deviations = [dev] if dev else []

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x, self.T)
