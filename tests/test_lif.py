"""The classical LIF reference (snnTorch ``Leaky``): its membrane update, its gradient path
and its size on the memprobe network.

Unlike the published QLIF (see ``test_qlif_ports.py``), the surrogate gradient of LIF
passes through emitted spikes and through the membrane, so both layers learn.
"""
from __future__ import annotations

import pytest
import torch

from sqs_bench.models.build import build_model
from sqs_bench.neurons import LIFLayer


def test_membrane_leaks_and_resets_by_subtraction():
    """One neuron with weight 1: U_t = 0.95 U_{t-1} + x_t - S_{t-1}, S_t = [U_t > 1]."""
    layer = LIFLayer(1, 1)
    mems, spikes = [], []
    with torch.no_grad():
        layer.fc.weight.fill_(1.0)
        state = layer.init_state(1, "cpu")
        for x in (0.6, 0.6, 0.6, 0.0):
            out, state, aux = layer(torch.tensor([[[x]]]), state)
            mems.append(float(aux["mem"]))
            spikes.append(float(out))
    # 0.6; 0.95*0.6 + 0.6 = 1.17 > 1 (spike); 0.95*1.17 + 0.6 - 1 = 0.7115; 0.95*0.7115
    assert mems == pytest.approx([0.6, 1.17, 0.7115, 0.675925])
    assert spikes == [0.0, 1.0, 0.0, 0.0]


def test_gradient_reaches_the_hidden_layer_and_earlier_steps():
    """A loss on the last output membrane only still trains the hidden layer, and reaches
    the input of the first step through the membrane."""
    torch.manual_seed(1)
    D, H, C, B, T = 20, 16, 4, 8, 4
    x = (torch.rand(T, B, 1, D) < 0.5).float().requires_grad_(True)
    l1, l2 = LIFLayer(D, H), LIFLayer(H, C)
    s1, s2 = l1.init_state(B, "cpu"), l2.init_state(B, "cpu")
    for t in range(T):
        o1, s1, _ = l1(x[t], s1)
        _, s2, aux2 = l2(o1.transpose(1, 2), s2)  # [B, H, 1] -> [B, 1, H]
    aux2["mem"].sum().backward()

    assert float(l1.fc.weight.grad.abs().sum()) > 0.0
    assert float(l2.fc.weight.grad.abs().sum()) > 0.0
    assert float(x.grad[0].abs().sum()) > 0.0


def test_param_count_of_the_memprobe_network(make_cfg):
    """One Linear plane without bias per layer: 5*64 + 64*4 = 576."""
    cfg = make_cfg(["model=lif"])
    model = build_model(cfg.model, in_features=5, n_classes=4, hidden=64)
    assert [layer.param_count() for layer in model.layers] == [320, 256]
    assert model.param_count() == 576
    assert all(isinstance(layer, LIFLayer) for layer in model.layers)
    assert model.layers[0].beta == 0.95 and model.layers[0].threshold == 1.0
