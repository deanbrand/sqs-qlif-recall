"""Shared interface for all spiking layers in sqs_bench.

Every neuron model (SQS, QLIF) is wrapped as a *layer*: a bank of ``out_features``
neurons fed by ``in_features`` presynaptic spike trains, stepped one time step at
a time. "The SQS paper" is Chen, Rajendran and Simeone, arXiv:2506.21324.

Tensor conventions (fixed; do not change without updating every implementer)
---------------------------------------------------------------------------
* ``x_t``      : ``[B, N_in, in_features]`` float -- presynaptic signals at time t.
                 ``N_in`` is the number of spike *channels* per presynaptic neuron
                 (the SQS paper's ``N``; 1 for QLIF).
* ``spikes``   : ``[B, out_features, N_out]`` float -- the layer's output signals.
                 Must be differentiable through whatever surrogate the neuron
                 defines (the value is hard 0/1).
* ``state``    : opaque per-layer recurrent state (membrane ``[B, out]`` for
                 QLIF, statevector ``[B, out, 2**(N+N_mem)]`` for SQS).
                 Created by ``init_state`` and threaded through ``forward``.
* ``aux``      : dict of extra tensors for losses / metrics.
                 Conventional keys (present when meaningful):
                   ``"mem"``     membrane potential ``[B, out]``  (QLIF)
                   ``"p"``       spike probability  ``[B, out, N_out]`` (SQS Born
                                 marginal P(s=1); for QLIF the multi-shot
                                 excitation probability alpha = mem)
                   ``"p_joint"`` ``[B, out, 2**N_out]`` (SQS only)
                   ``"outcome"`` ``[B, out]`` int64 joint-outcome index (SQS only)
                   ``"psi_in"``  statevector at the *start* of the step (SQS only)
                   ``"z"``       input currents ``[B, out, N_out]`` (SQS only)

Layers own their synaptic weights (the SQS paper's ``w``) as well as any neuron
parameters (the SQS PQC angles). ``FeedForwardSNN`` in ``sqs_bench.models`` just
chains layers and stacks records over time.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class SpikingLayer(nn.Module):
    """Abstract base class. Subclasses must set the attributes and implement
    ``init_state`` and ``forward``."""

    #: number of presynaptic input features (P in the SQS paper)
    in_features: int
    #: number of neurons in this layer (K)
    out_features: int
    #: number of input channels expected per presynaptic neuron (N_in)
    n_in_channels: int = 1
    #: number of output channels produced per neuron (N_out, the SQS paper's N)
    n_out_channels: int = 1

    def init_state(self, batch_size: int, device: torch.device | str) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x_t: Tensor, state: Any) -> tuple[Tensor, Any, dict[str, Tensor]]:  # pragma: no cover
        raise NotImplementedError

    def param_count(self) -> int:
        """Number of trainable parameters in this layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach_state(self, state: Any) -> Any:
        """Cut the BPTT graph through the recurrent state (truncated BPTT)."""
        if isinstance(state, Tensor):
            return state.detach()
        if isinstance(state, (tuple, list)):
            return type(state)(self.detach_state(s) for s in state)
        if isinstance(state, dict):
            return {k: self.detach_state(v) for k, v in state.items()}
        return state
