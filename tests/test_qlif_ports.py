"""The QLIF neuron must be bit-for-bit the published model.

``tests/reference/qsnn_alpha.py`` is the published ``alpha.py`` (MIT licence). Its
``Alpha`` / ``AlphaSurrogate`` are compared against the copy in
:mod:`sqs_bench.neurons.qlif` on source text, forward values and gradients. The
stacked-layer test then pins the published training behaviour: the backward pass
discards the spike gradient, so only the last layer learns.
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import pytest
import torch

from sqs_bench.models.build import build_model
from sqs_bench.neurons import QLIFLayer
from sqs_bench.neurons import qlif as ours

REFERENCE = Path(__file__).resolve().parent / "reference" / "qsnn_alpha.py"
PUBLISHED_URL = "https://github.com/deanbrand/QSNN/blob/main/alpha.py"


@pytest.fixture(scope="module")
def reference():
    name = "_qsnn_reference_alpha"
    spec = importlib.util.spec_from_file_location(name, REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # inspect.getsource looks the module up here
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def leaf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


# --------------------------------------------------------------------------- #
# provenance                                                                   #
# --------------------------------------------------------------------------- #
def test_provenance_recorded():
    assert ours._PROVENANCE == PUBLISHED_URL
    assert PUBLISHED_URL in REFERENCE.read_text(encoding="utf-8").splitlines()[0]


def test_original_classes_are_present():
    assert issubclass(ours.Alpha, torch.nn.Module)
    assert issubclass(ours.AlphaSurrogate, torch.autograd.Function)


def test_source_is_verbatim(reference):
    """Both classes are the published text, character for character."""
    assert inspect.getsource(ours.AlphaSurrogate) == inspect.getsource(reference.AlphaSurrogate)
    assert inspect.getsource(ours.Alpha) == inspect.getsource(reference.Alpha)


# --------------------------------------------------------------------------- #
# numerical equivalence                                                        #
# --------------------------------------------------------------------------- #
def run_steps(surrogate, thetas, taus, gates, mem0, threshold, T1):
    """Run T steps, returning (spikes, mems) and the leaf inputs for grad checks."""
    th = [leaf(t) for t in thetas]
    ta = [leaf(t) for t in taus]
    mem = mem0.clone()
    spikes, mems = [], []
    for theta, tau, gate in zip(th, ta, gates):
        spike, mem = surrogate.apply(theta, tau, gate, mem, threshold, T1)
        spikes.append(spike)
        mems.append(mem)
    return spikes, mems, th, ta


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_and_backward_match_the_reference(reference, seed):
    torch.manual_seed(seed)
    B, K, T = 5, 7, 6
    threshold, T1 = 0.75, 60.0

    thetas = [torch.rand(B, K) * math.pi for _ in range(T)]
    taus = [torch.rand(B, K) * 200.0 + 1.0 for _ in range(T)]
    gates = [(torch.rand(B, K) > 0.4).to(torch.int64) for _ in range(T)]
    mem0 = torch.zeros(B, K)

    a_spk, a_mem, a_th, a_ta = run_steps(
        reference.AlphaSurrogate, thetas, taus, gates, mem0, threshold, T1
    )
    b_spk, b_mem, b_th, b_ta = run_steps(
        ours.AlphaSurrogate, thetas, taus, gates, mem0, threshold, T1
    )

    for t in range(T):
        assert torch.equal(a_spk[t], b_spk[t]), f"spike mismatch at step {t}"
        assert torch.equal(a_mem[t], b_mem[t]), f"mem mismatch at step {t}"

    # a loss that touches both outputs, so the discarded grad_spike_out is exercised
    weights = [torch.rand(B, K) for _ in range(T)]
    loss_a = sum((m * w).sum() + s.sum() for m, s, w in zip(a_mem, a_spk, weights))
    loss_b = sum((m * w).sum() + s.sum() for m, s, w in zip(b_mem, b_spk, weights))
    assert torch.equal(loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()

    for t in range(T):
        assert torch.equal(a_th[t].grad, b_th[t].grad), f"grad theta mismatch at step {t}"
        assert torch.equal(a_ta[t].grad, b_ta[t].grad), f"grad tau mismatch at step {t}"
    # sanity: the reference gradients are not trivially zero everywhere
    assert any(float(g.grad.abs().sum()) > 0 for g in a_th)


def test_alpha_module_matches_the_reference(reference):
    """The module wrapper (state handling, reset_mem) must agree too."""
    torch.manual_seed(9)
    B, K, T = 3, 4, 5
    a = reference.Alpha(N=K)
    b = ours.Alpha(N=K)
    a.reset_mem()
    b.reset_mem()
    for _ in range(T):
        theta = torch.rand(B, K) * math.pi
        tau = torch.rand(B, K) * 200.0 + 1.0
        gate = (torch.rand(B, K) > 0.5).to(torch.int64)
        spk_a, mem_a = a(theta, tau, gate)
        spk_b, mem_b = b(theta, tau, gate)
        assert torch.equal(spk_a, spk_b)
        assert torch.equal(mem_a, mem_b)


def test_layer_runs_the_reference_function(reference):
    """``QLIFLayer.forward`` is the published wiring around the published function: rebuild
    the call by hand with the reference ``AlphaSurrogate`` and get the same numbers."""
    torch.manual_seed(3)
    B, D, K, T = 6, 5, 4, 4
    layer = QLIFLayer(D, K)
    x = (torch.rand(T, B, 1, D) < 0.5).float()
    state = layer.init_state(B, "cpu")
    mem_ref = torch.zeros(B, K)
    for t in range(T):
        spikes, state, aux = layer(x[t], state)
        lin = layer.fcq(x[t][:, 0])
        theta = torch.abs(lin) * torch.pi
        tau = torch.abs(layer.fct(x[t][:, 0])) * 200.0
        gate = torch.where(lin > 0, torch.tensor(1), torch.tensor(0))
        spk_ref, mem_ref = reference.AlphaSurrogate.apply(theta, tau, gate, mem_ref, 0.75, 60.0)
        assert torch.equal(spikes[..., 0], spk_ref), t
        assert torch.equal(state, mem_ref), t
        assert torch.equal(aux["mem"], mem_ref) and torch.equal(aux["p"][..., 0], mem_ref)


# --------------------------------------------------------------------------- #
# stacked-layer gradient behaviour (documents the published model)             #
# --------------------------------------------------------------------------- #
def stack_and_backward(D=20, H=16, C=4, B=8, T=4, seed=1):
    """Two layers, loss on the output membrane. Returns (l1, l2, hidden_spike_count)."""
    torch.manual_seed(seed)
    x = (torch.rand(T, B, 1, D) < 0.5).float()
    l1, l2 = QLIFLayer(D, H), QLIFLayer(H, C)
    s1, s2 = l1.init_state(B, "cpu"), l2.init_state(B, "cpu")
    mems, hidden = [], 0.0
    for t in range(T):
        o1, s1, _ = l1(x[t], s1)
        hidden += float(o1.detach().sum())
        o2, s2, aux2 = l2(o1.transpose(1, 2), s2)  # [B, H, 1] -> [B, 1, H]
        mems.append(aux2["mem"])
    torch.stack(mems).sum().backward()
    return l1, l2, hidden


def grad_magnitude(param):
    return 0.0 if param.grad is None else float(param.grad.abs().sum())


def test_hidden_layer_gets_no_gradient():
    """Published behaviour: backward() drops grad_spike_out, so nothing upstream
    of the last layer learns. This is a regression guard, not an endorsement."""
    l1, l2, hidden_spikes = stack_and_backward()
    assert hidden_spikes > 0, "hidden layer never spiked; test is vacuous"

    for name, p in l1.named_parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, f"hidden {name} got gradient"

    assert grad_magnitude(l2.fcq.weight) > 0.0
    assert grad_magnitude(l2.fcq.bias) > 0.0
    assert grad_magnitude(l2.fct.weight) > 0.0


def test_layer_smoke_three_steps():
    torch.manual_seed(0)
    B, T = 6, 3
    layer = QLIFLayer(12, 5)
    assert layer.in_features == 12 and layer.out_features == 5
    assert layer.n_in_channels == 1 and layer.n_out_channels == 1

    state = layer.init_state(B, "cpu")
    assert state.shape == (B, 5)

    x = (torch.rand(T, B, 1, 12) < 0.5).float()
    for t in range(T):
        out, state, aux = layer(x[t], state)
        assert out.shape == (B, 5, 1)
        assert torch.isfinite(out).all()
        assert aux["mem"].shape == (B, 5)
        assert aux["p"].shape == (B, 5, 1)
        assert torch.isfinite(aux["mem"]).all()
        assert set(out.unique().tolist()) <= {0.0, 1.0}
        state = layer.detach_state(state)
        assert not state.requires_grad


def test_param_count_of_the_memprobe_network(make_cfg):
    """Two Linear planes with bias per layer: 2 * (5*64 + 64) + 2 * (64*4 + 4) = 1288."""
    cfg = make_cfg(["model=qlif"])
    model = build_model(cfg.model, in_features=5, n_classes=4, hidden=64)
    assert [layer.param_count() for layer in model.layers] == [768, 520]
    assert model.param_count() == 1288
    assert all(isinstance(layer, QLIFLayer) for layer in model.layers)
    assert model.layers[0].tau_scale == 200.0 and model.layers[0].threshold == 0.75
