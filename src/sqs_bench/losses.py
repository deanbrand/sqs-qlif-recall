"""Training objective of the ``sqs_ce`` protocol.

Cross entropy on the output membrane (the Born spiking probability for an SQS output
layer), summed over the scored time steps and mean-reduced over the batch.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def membrane_ce_loss(mem_seq: Tensor, targets: Tensor, window: int | None = None) -> Tensor:
    """Cross entropy on the output membrane, summed over time.

    ``mem_seq`` is ``[T, B, C]`` (the QLIF reference code's ``mem_rec``); the loss is
    ``sum_t CE(mem_t, y)``, mean-reduced over the batch at every step. With ``window``
    the sum runs over the last ``window`` steps only.
    """
    if mem_seq.dim() == 4 and mem_seq.shape[-1] == 1:
        mem_seq = mem_seq.squeeze(-1)
    if mem_seq.dim() != 3:
        raise ValueError(f"mem_seq must be [T,B,C], got {tuple(mem_seq.shape)}")
    if window:
        from sqs_bench.metrics import window_slice
        mem_seq = window_slice(mem_seq, window)
    y = targets.long()
    total = mem_seq.new_zeros(())
    for t in range(mem_seq.shape[0]):
        total = total + F.cross_entropy(mem_seq[t], y)
    return total
