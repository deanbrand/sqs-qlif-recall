"""QLIF -- the published two-weight-plane quantum leaky integrate-and-fire neuron.

``Alpha`` and ``AlphaSurrogate`` below are copied VERBATIM from the published
model (see ``_PROVENANCE``; MIT licence). Every quirk is deliberate and must survive:

* ``backward`` discards ``grad_spike_out``, so no gradient flows through the
  emitted spike. In a stacked network only the *last* layer is trained.
* ``grad_mem`` uses an in-place ``pow_`` on a temporary.
* the returned gradient tuple puts ``grad_mem`` in the ``x`` slot and ``None``
  in the ``mem`` slot, so there is no BPTT through the membrane either.

:class:`QLIFLayer` wraps that function in the project's
:class:`~sqs_bench.neurons.base.SpikingLayer` interface, wiring the two linear
planes exactly as the published notebook in the same repository does.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import SpikingLayer

#: source of the verbatim copy below (MIT licence)
_PROVENANCE = "https://github.com/deanbrand/QSNN/blob/main/alpha.py"


# --------------------------------------------------------------------------- #
# VERBATIM COPY -- do not reformat, do not "fix"                               #
# --------------------------------------------------------------------------- #
class Alpha(nn.Module):
    def __init__(self, threshold=0.75, T1=60, N=1, device=torch.device("cpu")):
        super(Alpha, self).__init__()

        self.mem = torch.zeros(N, dtype=torch.float32)
        self.threshold = threshold
        self.T1 = T1
        self.device = device

    def forward(self, theta, tau, x):
        theta = theta.to(self.device)
        tau = tau.to(self.device)
        x = x.to(self.device)

        spike_out, self.mem = AlphaSurrogate.apply(theta, tau, x, self.mem,
                                                   self.threshold, self.T1)
        return spike_out, self.mem

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.device)
        return self.mem


class AlphaSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, tau, x, mem, threshold, T1):
        ctx.save_for_backward(theta, tau, x, mem)
        ctx.threshold = threshold
        ctx.T1 = T1

        gamma = -2*torch.arcsin(torch.sqrt(mem * torch.exp(-tau/T1)))
        phi = torch.where(mem < threshold,
                          2 * torch.arcsin(torch.sqrt(mem)), 0.0)

        Theta = torch.sin((theta + phi) * x/2)**2
        Gamma = torch.sin((gamma + phi) * (1 - x)/2)**2
        mem = torch.where(mem < threshold, Gamma + Theta, 0.0)

        spike_out = torch.where(mem > threshold,
                                1.0, 0.0)

        mem = torch.where(mem > 1, 0.0, mem)

        return spike_out, mem

    @staticmethod
    def backward(ctx, grad_spike_out, grad_output):
        theta, tau, x, mem = ctx.saved_tensors
        T1 = ctx.T1
        grad_mem = 1 / (1 + (torch.pi * mem).pow_(2)) * grad_output

        sqrt_mem = torch.sqrt(mem)
        sqrt_mem_exp = torch.sqrt(mem * torch.exp(-tau/T1))

        dtheta = (x * torch.sin(x * (theta + 2 * torch.arcsin(sqrt_mem)))) / 2
        dtau = ((x - 1) * sqrt_mem_exp * torch.sin(2 * (x - 1) * (torch.arcsin(sqrt_mem) -
                torch.arcsin(sqrt_mem_exp)))) / (2 * T1 * torch.sqrt(1 - mem * torch.exp(-tau/T1)))

        grad_theta = (dtheta * grad_mem)
        grad_tau = (dtau * grad_mem)

        return grad_theta, grad_tau, grad_mem, None, None, None
# --------------------------------------------------------------------------- #
# END VERBATIM COPY                                                            #
# --------------------------------------------------------------------------- #


class QLIFLayer(SpikingLayer):
    """A bank of ``out_features`` published QLIF neurons.

    Wiring (from the published notebook)::

        lin    = fcq(x)
        theta  = |lin| * pi
        tau    = |fct(x)| * tau_scale
        x_gate = (lin > 0)                      # int64, as published
        spike, mem = AlphaSurrogate(theta, tau, x_gate, mem, threshold, T1)

    ``tau_scale`` is the time-constant scale ``S``; every run here uses 200.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        threshold: float = 0.75,
        T1: float = 60.0,
        tau_scale: float = 200.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_in_channels = 1
        self.n_out_channels = 1

        self.threshold = float(threshold)
        self.T1 = float(T1)
        self.tau_scale = float(tau_scale)

        self.fcq = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.fct = nn.Linear(self.in_features, self.out_features, bias=bias)

    def init_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        # the published code starts from zeros(N) and lets broadcasting do the rest;
        # materialising [B, out] is numerically identical.
        return torch.zeros(batch_size, self.out_features, dtype=torch.float32, device=device)

    def forward(self, x_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        x = x_t[:, 0]  # [B, in_features]

        lin = self.fcq(x)
        theta = torch.abs(lin) * torch.pi
        tau = torch.abs(self.fct(x)) * self.tau_scale
        x_gate = torch.where(lin > 0, torch.tensor(1), torch.tensor(0))

        spike, mem = AlphaSurrogate.apply(theta, tau, x_gate, state, self.threshold, self.T1)

        aux = {"mem": mem, "p": mem.unsqueeze(-1)}
        return spike.unsqueeze(-1), mem, aux

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"threshold={self.threshold}, T1={self.T1}, tau_scale={self.tau_scale}"
        )
