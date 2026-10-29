"""SQS layer bookkeeping: parameter counts, the embedding kink, and the autocast guard."""
from __future__ import annotations

import pytest
import torch

from sqs_bench.neurons.sqs import _TWO_LAYER_ANSATZE, ANSATZE, SQSLayer, pqc_param_count


def test_the_two_ansatze():
    assert ANSATZE == ("crx_all", "crx_both")
    assert _TWO_LAYER_ANSATZE == ("crx_both",)


def test_param_count_memprobe_network():
    """The memprobe network 5 -> 64 -> 4 at N = N_mem = 1: weights plus 3 angles per neuron."""
    hidden = SQSLayer(5, 64, 1, 1)
    out = SQSLayer(64, 4, 1, 1)
    assert hidden.param_count() == 5 * 64 + 3 * 64 == 512
    assert out.param_count() == 64 * 4 + 3 * 4 == 268
    assert hidden.weight.shape == (1, 5, 64)
    assert hidden.theta_crx.shape == (64, 1, 1)
    assert hidden.theta_rx.shape == (64, 2)
    assert hidden.theta_crx2 is None


@pytest.mark.parametrize(
    "ansatz,n_io,n_mem,g",
    [
        ("crx_all", 1, 1, 3),
        ("crx_all", 2, 1, 5),
        ("crx_all", 2, 2, 8),
        ("crx_both", 1, 1, 4),
        ("crx_both", 2, 1, 7),
        ("crx_both", 2, 2, 12),
    ],
)
def test_pqc_parameter_count(ansatz, n_io, n_mem, g):
    assert pqc_param_count(n_io, n_mem, ansatz) == g
    k, p = 6, 4
    layer = SQSLayer(p, k, n_io, n_mem, ansatz=ansatz)
    assert layer.G == g
    pqc = sum(
        t.numel() for t in (layer.theta_crx, layer.theta_crx2, layer.theta_rx) if t is not None
    )
    assert pqc == g * k
    assert layer.param_count() == n_io * p * k + g * k
    assert (layer.theta_crx2 is not None) == (ansatz in _TWO_LAYER_ANSATZE)


def test_pqc_param_count_rejects_unknown_ansatz():
    with pytest.raises(ValueError):
        pqc_param_count(1, 1, "strongly_entangling")


def test_zero_current_gives_identity_embedding():
    """z <= 0 must apply no rotation at all (paper Eq. (11))."""
    torch.manual_seed(0)
    layer = SQSLayer(3, 2, 1, 1, dtype=torch.complex128)
    with torch.no_grad():
        layer.weight.fill_(-1.0)
    x = torch.ones(1, 1, 3, dtype=torch.float64)
    z = layer.currents(x)
    assert torch.all(z < 0)
    assert torch.all(layer.angles(z) == 0)


def test_autocast_is_rejected():
    layer = SQSLayer(3, 2, 1, 1)
    x = torch.rand(1, 1, 3)
    psi = layer.init_state(1, "cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            layer(x, psi)
