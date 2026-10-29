"""Spike encoders: turn a data batch into the ``[T, B, D]`` train the network consumes.

The memory probe is already a spike train, so the only encoder is :func:`identity`,
which moves the time axis to the front.
"""
from __future__ import annotations

from torch import Tensor

__all__ = ["identity"]


def identity(x: Tensor, T: int) -> Tensor:
    """Pass a sequence through: ``[B, T, D]`` -> ``[T, B, D]``.

    For data that is already temporal (the memory probe): there is nothing to
    encode, only an axis to move. Raises when the sequence length does not match
    the ``T`` the protocol resolved, because silently truncating or padding a
    spike train would change the task.
    """
    if x.dim() != 3:
        raise ValueError(f"identity expects [B, T, D], got shape {tuple(x.shape)}")
    if int(x.shape[1]) != int(T):
        raise ValueError(
            f"identity: the data has {int(x.shape[1])} time steps but T={int(T)}; set "
            "dataset.T (or protocol.T) to the length of the sequence"
        )
    return x.permute(1, 0, 2).contiguous()
