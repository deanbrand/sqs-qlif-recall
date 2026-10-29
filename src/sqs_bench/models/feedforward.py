"""Feed-forward spiking network: a chain of :class:`~sqs_bench.neurons.base.SpikingLayer`.

The network is stepped over time; every layer keeps its own recurrent state.
Tensor conventions follow ``neurons/base.py``:

* a layer consumes ``[B, N_in, in_features]`` and emits ``[B, out_features, N_out]``,
  so between two layers the channel axis moves from the back to the front
  (``spikes.transpose(1, 2)``).
* the network input is ``[T, B, N, D]`` (or ``[T, B, D]``, treated as ``N = 1``).

Everything a loss or a metric needs comes back in :class:`ForwardOut`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from sqs_bench.neurons.base import SpikingLayer


@dataclass
class ForwardOut:
    """Everything the training loop and the metrics need from one sequence."""

    #: output-layer spikes, ``[T, B, C, N]``
    spikes_out: Tensor
    #: mean over time of ``spikes_out``, ``[B, C, N]`` -- differentiable
    rate_out: Tensor
    #: output-layer membrane, ``[T, B, C]`` (``None`` if the layer has no ``"mem"``)
    mem_out: Tensor | None
    #: output-layer spiking probability, ``[T, B, C, N]`` (``None`` if no ``"p"``)
    p_out: Tensor | None
    #: hidden-layer spikes, one ``[T, B, H, N]`` per hidden layer (empty unless
    #: ``record_hidden``)
    spikes_hid: list[Tensor] = field(default_factory=list)
    #: hidden-layer ``"p"``, one ``[T, B, H, N]`` per hidden layer, or ``None``
    p_hid: list[Tensor] | None = None
    #: hidden-layer ``"mem"``, one ``[T, B, H]`` per hidden layer, or ``None``
    mem_hid: list[Tensor] | None = None
    #: total spikes emitted by the whole network (hidden + output) per time step
    #: and per sample, ``[T, B]``; detached, from hard spike values
    n_spikes: Tensor | None = None
    #: the same count split per layer: one ``[T, B]`` per layer (hidden first)
    n_spikes_layers: list[Tensor] = field(default_factory=list)
    #: ``[layer][t] -> aux dict`` -- only filled when ``record_aux=True``
    layer_aux: list[list[dict[str, Tensor]]] | None = None


def _stack(seq: list[Tensor] | None) -> Tensor | None:
    if not seq:
        return None
    return torch.stack(seq, dim=0)


def _spike_count(layer: SpikingLayer, spikes: Tensor, aux: dict[str, Tensor]) -> Tensor:
    """Hard spikes emitted by ``layer`` at this step, ``[B]`` (detached).

    Binary spikes are simply summed. A layer can override this by putting
    ``"n_spikes"`` (``[B]``) or ``"spk"`` (spike-shaped) in its aux dict.
    """
    if "n_spikes" in aux:
        return aux["n_spikes"].detach().reshape(spikes.shape[0], -1).sum(-1)
    if "spk" in aux:
        return aux["spk"].detach().flatten(1).sum(-1)
    s = spikes.detach()
    return s.flatten(1).sum(-1)


class FeedForwardSNN(nn.Module):
    """Chain of spiking layers, stepped over ``T`` time steps.

    Parameters
    ----------
    layers:
        Ordered list of :class:`SpikingLayer`; the last one is the output layer.
    detach_state:
        Truncated BPTT -- detach every layer state after each step.
    record_hidden:
        Stack hidden-layer spikes / ``p`` / ``mem`` into :class:`ForwardOut`.
        Spike *counts* are recorded either way (they are cheap).
    record_aux:
        Keep every layer's aux dict for every step. Memory hungry; off by default.
    """

    def __init__(
        self,
        layers: Sequence[SpikingLayer],
        detach_state: bool = False,
        record_hidden: bool = True,
        record_aux: bool = False,
    ) -> None:
        super().__init__()
        if len(layers) == 0:
            raise ValueError("FeedForwardSNN needs at least one layer")
        self.layers = nn.ModuleList(layers)
        self.detach_state = bool(detach_state)
        self.record_hidden = bool(record_hidden)
        self.record_aux = bool(record_aux)

    # ---------------------------------------------------------------- helpers
    @property
    def in_features(self) -> int:
        return int(self.layers[0].in_features)

    @property
    def out_features(self) -> int:
        return int(self.layers[-1].out_features)

    @property
    def n_in_channels(self) -> int:
        """Channels the *network* expects on its input (``N`` of the first layer)."""
        return int(getattr(self.layers[0], "n_in_channels", 1))

    def units_per_layer(self) -> list[int]:
        """``out_features * n_out_channels`` per layer -- the denominator of the
        per-neuron spike rate (SQS paper Fig. 10b counts spike channels)."""
        return [int(l.out_features) * int(getattr(l, "n_out_channels", 1)) for l in self.layers]

    def total_units(self) -> int:
        return int(sum(self.units_per_layer()))

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def init_state(self, batch_size: int, device: torch.device | str) -> list[Any]:
        return [l.init_state(batch_size, device) for l in self.layers]

    @staticmethod
    def _match_channels(x: Tensor, n_in: int, idx: int) -> Tensor:
        """Make ``x`` ``[B, C, F]`` have ``C == n_in`` by broadcasting ``C == 1``."""
        c = x.shape[1]
        if c == n_in:
            return x
        if c == 1:
            return x.expand(x.shape[0], n_in, x.shape[2])
        raise ValueError(
            f"layer {idx} expects {n_in} input channel(s) but got {c}; only a "
            "single channel can be broadcast"
        )

    # ---------------------------------------------------------------- forward
    def forward(self, x_seq: Tensor) -> ForwardOut:
        if x_seq.dim() == 3:  # [T, B, D] -> [T, B, 1, D]
            x_seq = x_seq.unsqueeze(2)
        if x_seq.dim() != 4:
            raise ValueError(f"expected x_seq [T,B,N,D] or [T,B,D], got {tuple(x_seq.shape)}")
        T, B = x_seq.shape[0], x_seq.shape[1]
        device = x_seq.device
        n_layers = len(self.layers)

        states = self.init_state(B, device)
        spikes_rec: list[list[Tensor]] = [[] for _ in range(n_layers)]
        p_rec: list[list[Tensor]] = [[] for _ in range(n_layers)]
        mem_rec: list[list[Tensor]] = [[] for _ in range(n_layers)]
        count_rec: list[list[Tensor]] = [[] for _ in range(n_layers)]
        aux_rec: list[list[dict[str, Tensor]]] = [[] for _ in range(n_layers)]

        for t in range(T):
            cur = x_seq[t]  # [B, N, D]
            for i, layer in enumerate(self.layers):
                if cur.shape[2] != layer.in_features:
                    raise ValueError(
                        f"layer {i} expects in_features={layer.in_features} but got "
                        f"{cur.shape[2]}"
                    )
                cur = self._match_channels(cur, int(getattr(layer, "n_in_channels", 1)), i)
                spikes, states[i], aux = layer(cur, states[i])
                if self.detach_state:
                    states[i] = layer.detach_state(states[i])

                count_rec[i].append(_spike_count(layer, spikes, aux))
                if i == n_layers - 1 or self.record_hidden:
                    spikes_rec[i].append(spikes)
                    if "p" in aux:
                        p_rec[i].append(aux["p"])
                    if "mem" in aux:
                        mem_rec[i].append(aux["mem"])
                if self.record_aux:
                    aux_rec[i].append(aux)

                cur = spikes.transpose(1, 2)  # [B, out, N_out] -> [B, N_out, out]

        spikes_out = torch.stack(spikes_rec[-1], dim=0)  # [T, B, C, N]
        rate_out = spikes_out.mean(dim=0)
        p_out = _stack(p_rec[-1])
        mem_out = _stack(mem_rec[-1])

        spikes_hid = [torch.stack(s, dim=0) for s in spikes_rec[:-1] if s]
        p_hid = [torch.stack(p, dim=0) for p in p_rec[:-1] if p] or None
        mem_hid = [torch.stack(m, dim=0) for m in mem_rec[:-1] if m] or None

        n_spikes_layers = [torch.stack(c, dim=0) for c in count_rec]  # each [T, B]
        n_spikes = torch.stack(n_spikes_layers, dim=0).sum(dim=0)  # [T, B]

        return ForwardOut(
            spikes_out=spikes_out,
            rate_out=rate_out,
            mem_out=mem_out,
            p_out=p_out,
            spikes_hid=spikes_hid,
            p_hid=p_hid,
            mem_hid=mem_hid,
            n_spikes=n_spikes,
            n_spikes_layers=n_spikes_layers,
            layer_aux=aux_rec if self.record_aux else None,
        )
