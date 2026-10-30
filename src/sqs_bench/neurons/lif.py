"""LIF -- the classical leaky integrate-and-fire neuron, as implemented by snnTorch.

This is the classical reference for the recall study. It keeps a membrane
potential ``U`` from one step to the next, like QLIF does::

    U_t = beta * U_{t-1} + W x_t - theta * S_{t-1}      # leak, input, reset
    S_t = H(U_t - theta)

The reset subtracts the threshold one step after a spike (snnTorch's default).
The spike's gradient is the fast-sigmoid surrogate, so, unlike the published
QLIF, the gradient reaches every layer and every earlier time step.

Reference: Eshraghian et al., "Training spiking neural networks using lessons
from deep learning", Proc. IEEE 111, 1016 (2023).
"""
from __future__ import annotations

import snntorch as snn
import torch
from snntorch import surrogate
from torch import Tensor, nn

from .base import SpikingLayer


class LIFLayer(SpikingLayer):
    """``Linear -> snnTorch Leaky``: a bank of ``out_features`` LIF neurons.

    ``bias=False`` (the default) gives exactly ``in_features * out_features``
    parameters. The membrane is passed in and out explicitly
    (``init_hidden=False``), as for every other layer here.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        beta: float = 0.95,
        threshold: float = 1.0,
        slope: float = 25.0,
        reset_mechanism: str = "subtract",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_in_channels = 1
        self.n_out_channels = 1

        self.beta = float(beta)
        self.threshold = float(threshold)
        self.slope = float(slope)
        self.reset_mechanism = reset_mechanism

        self.fc = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=surrogate.fast_sigmoid(slope=self.slope),
            init_hidden=False,
            reset_mechanism=reset_mechanism,
        )

    def init_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, self.out_features, dtype=torch.float32, device=device)

    def forward(self, x_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        cur = self.fc(x_t[:, 0])  # [B, out_features]
        spk, mem = self.lif(cur, state)
        return spk.unsqueeze(-1), mem, {"mem": mem}

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"beta={self.beta}, threshold={self.threshold}, slope={self.slope}, "
            f"reset_mechanism={self.reset_mechanism!r}"
        )
