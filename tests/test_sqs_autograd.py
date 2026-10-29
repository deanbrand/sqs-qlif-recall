"""Autograd: exact gradients of the likelihood, and the Appendix-A surrogate slope."""
from __future__ import annotations

import itertools
import math

import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from sqs_bench.neurons.sqs import ANSATZE, SQSLayer

CASES = [(1, 1), (2, 1), (2, 2)]
GRADCHECK_CASES = [
    ("crx_all", 1, 1),
    ("crx_all", 2, 1),
    ("crx_all", 2, 2),
    ("crx_both", 1, 1),
    ("crx_both", 2, 2),
]
#: per-ansatz seed offsets for the gradcheck layers; the guard that keeps every input current
#: away from the kink of Eq. (11) was checked at exactly these seeds
SEED_OFFSET = {"crx_all": 0, "crx_both": 6}


def _free_params(layer):
    """The layer's parameters as independent float64 leaves, keyed for
    ``functional_call`` on a wrapper module whose submodule is ``layer``."""
    return {
        "layer." + name: p.detach().clone().requires_grad_(True)
        for name, p in layer.named_parameters()
    }


class _Unrolled(torch.nn.Module):
    """Sum of ``log p(outcome_t)`` over ``T`` teacher-forced steps.

    Uses only :meth:`SQSLayer.forward_with_outcome`, so the whole thing is a
    smooth function of the parameters (no sampling, no straight-through).
    """

    def __init__(self, layer: SQSLayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, xs: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        psi = self.layer.init_state(xs.shape[1], xs.device)
        total = xs.new_zeros(())
        for t in range(xs.shape[0]):
            p_sel, psi = self.layer.forward_with_outcome(xs[t], psi, outcomes[t])
            total = total + torch.log(p_sel).sum()
        return total


def _greedy_outcomes(layer, xs):
    """Freeze a high-probability outcome per step so ``log p`` stays well conditioned."""
    outcomes = []
    with torch.no_grad():
        psi = layer.init_state(xs.shape[1], xs.device)
        for t in range(xs.shape[0]):
            p_joint, _ = layer.probabilities(xs[t], psi)
            j = p_joint.argmax(-1)
            _, psi = layer.forward_with_outcome(xs[t], psi, j)
            outcomes.append(j)
    return torch.stack(outcomes)


@pytest.mark.parametrize("ansatz,n_io,n_mem", GRADCHECK_CASES)
def test_gradcheck_log_likelihood(ansatz, n_io, n_mem):
    torch.manual_seed(31 + n_io + 5 * n_mem + SEED_OFFSET[ansatz])
    t_steps, b, k, p = 3, 2, 2, 3
    layer = SQSLayer(
        p, k, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128, theta_init="uniform"
    )
    xs = 0.5 + 0.5 * torch.rand(t_steps, b, n_io, p, dtype=torch.float64)

    # Keep every |z| well away from the kink of Eq. (11), so the function is
    # smooth in the finite-difference neighbourhood gradcheck probes. Channel 0
    # gets z > 0 (the RX branch), channel 1 (when it exists) gets z < 0 (the
    # "no rotation" branch, whose weight gradient must be exactly zero).
    with torch.no_grad():
        layer.weight.abs_().add_(0.1)
        if n_io > 1:
            layer.weight[1].neg_()
        z = torch.stack([layer.currents(xs[t]) for t in range(t_steps)])
        assert z.abs().min() > 1e-2, f"input current too close to the phi kink: {z.abs().min()}"

    outcomes = _greedy_outcomes(layer, xs)
    model = _Unrolled(layer)
    free = _free_params(layer)
    names = list(free)
    assert ("layer.theta_crx2" in names) == (ansatz == "crx_both")

    def f(*values):
        return functional_call(model, dict(zip(names, values)), (xs, outcomes))

    values = tuple(free[n] for n in names)
    assert gradcheck(f, values, eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True)

    f(*values).backward()
    for name, value in zip(names, values):
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
    w = free["layer.weight"]
    if n_io > 1:  # the z <= 0 branch contributes no weight gradient
        assert torch.count_nonzero(w.grad[1]) == 0
        assert torch.count_nonzero(w.grad[0]) > 0


def test_gradcheck_marginal_probability():
    """Gradients of the Born marginal itself (what the surrogate multiplies)."""
    torch.manual_seed(8)
    layer = SQSLayer(3, 2, 2, 1, dtype=torch.complex128)
    x = torch.rand(2, 2, 3, dtype=torch.float64)
    psi = layer.init_state(2, "cpu")

    class _Probs(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x, psi):
            p_joint, _ = self.layer.probabilities(x, psi)
            return (p_joint * torch.arange(1, p_joint.shape[-1] + 1, dtype=p_joint.dtype)).sum()

    model = _Probs(layer)
    w = layer.weight.detach().clone().requires_grad_(True)
    tc = layer.theta_crx.detach().clone().requires_grad_(True)
    tr = layer.theta_rx.detach().clone().requires_grad_(True)

    def f(w_, tc_, tr_):
        params = {"layer.weight": w_, "layer.theta_crx": tc_, "layer.theta_rx": tr_}
        return functional_call(model, params, (x, psi))

    assert gradcheck(f, (w, tc, tr), eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True)


@pytest.mark.parametrize("surrogate", ["marginal_sigmoid", "ste", "joint_sigmoid"])
def test_surrogate_slope(surrogate):
    """ds/dp is sigma'(p) for the sigmoid surrogates and 1 for the plain STE."""
    layer = SQSLayer(3, 4, 2, 1, surrogate=surrogate, dtype=torch.complex128)
    p = torch.rand(5, 4, 2, dtype=torch.float64, requires_grad=True)
    p_sel = torch.rand(5, 4, dtype=torch.float64, requires_grad=True)
    s_hard = torch.randint(0, 2, (5, 4, 2)).double()

    s = layer.surrogate_spikes(s_hard, p, p_sel)
    # forward value is the hard sample, always
    torch.testing.assert_close(s.detach(), s_hard)

    g = torch.ones_like(s)
    if surrogate == "joint_sigmoid":
        (grad_sel,) = torch.autograd.grad(s, p_sel, g)
        sig = torch.sigmoid(p_sel)
        torch.testing.assert_close(grad_sel, sig * (1 - sig) * s.shape[-1])
        assert torch.autograd.grad(s, p, g, allow_unused=True)[0] is None
    else:
        (grad_p,) = torch.autograd.grad(s, p, g)
        if surrogate == "ste":
            torch.testing.assert_close(grad_p, torch.ones_like(p))
        else:
            sig = torch.sigmoid(p)
            torch.testing.assert_close(grad_p, sig * (1 - sig))


def test_surrogate_slope_numerically_through_forward():
    """Same check end-to-end: perturbing the Born marginal by dp changes the
    backward-pass spike value by sigma'(p) dp."""
    torch.manual_seed(0)
    layer = SQSLayer(3, 2, 1, 1, dtype=torch.complex128)
    x = torch.rand(4, 1, 3, dtype=torch.float64)
    psi = layer.init_state(4, "cpu")
    spikes, _, aux = layer(x, psi)
    p = aux["p"]
    (grad_p,) = torch.autograd.grad(spikes.sum(), p, retain_graph=True)
    sig = torch.sigmoid(p.detach())
    torch.testing.assert_close(grad_p, sig * (1 - sig))


def test_gradients_reach_all_parameters():
    torch.manual_seed(1)
    layer = SQSLayer(5, 3, 2, 2)
    x = torch.rand(4, 2, 5)
    psi = layer.init_state(4, "cpu")
    total = 0.0
    for _ in range(3):
        spikes, psi, aux = layer(x, psi)
        total = total + spikes.sum() + torch.log(aux["p_sel"].clamp_min(1e-8)).sum()
    total.backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_detach_state_cuts_cross_time_gradient():
    """``detach_state`` must remove the BPTT path; the layer must not detach by
    itself.

    ``N_mem = 2`` is required for this test to have any content: with a single
    memory qubit the measurement statistics do not depend on the memory state at
    all (see ``test_memory_state_does_not_affect_output_when_nmem_is_1``), so the
    cross-time gradient is exactly zero whether or not the state is detached.
    """
    torch.manual_seed(17)
    layer = SQSLayer(4, 2, 1, 2, dtype=torch.complex128)
    with torch.no_grad():  # positive currents, so the CRX is actually driven
        layer.weight.abs_().add_(0.1)
    x1 = 0.5 + 0.5 * torch.rand(3, 1, 4, dtype=torch.float64)
    x2 = 0.5 + 0.5 * torch.rand(3, 1, 4, dtype=torch.float64)
    outcome = torch.zeros(3, 2, dtype=torch.int64)

    def run(detach: bool):
        layer.zero_grad(set_to_none=True)
        psi = layer.init_state(3, "cpu")
        _, psi = layer.forward_with_outcome(x1, psi, outcome)
        if detach:
            psi = layer.detach_state(psi)
        p_sel, _ = layer.forward_with_outcome(x2, psi, outcome)
        torch.log(p_sel).sum().backward()
        return {n: p.grad.clone() for n, p in layer.named_parameters()}

    full = run(False)
    cut = run(True)
    # the truncated gradient must equal the one obtained by starting the second
    # step from a constant state
    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        psi = layer.init_state(3, "cpu")
        _, psi_const = layer.forward_with_outcome(x1, psi, outcome)
    p_sel, _ = layer.forward_with_outcome(x2, psi_const, outcome)
    torch.log(p_sel).sum().backward()
    ref = {n: p.grad.clone() for n, p in layer.named_parameters()}

    for name in ref:
        torch.testing.assert_close(cut[name], ref[name])
    assert not torch.allclose(full["theta_crx"], cut["theta_crx"]), (
        "BPTT through the memory qubit produced no extra gradient"
    )
    assert not torch.allclose(full["weight"], cut["weight"])


def test_no_nan_when_amplitudes_vanish():
    """|0...0> has exactly-zero amplitudes; ``abs()`` would give NaN gradients."""
    layer = SQSLayer(2, 1, 1, 1, dtype=torch.complex128)
    with torch.no_grad():
        layer.weight.zero_()  # z = 0 -> phi = 0
        layer.theta_crx.zero_()
        layer.theta_rx.zero_()  # U = I, so psi stays |00> and 3 amplitudes are 0
    x = torch.zeros(2, 1, 2, dtype=torch.float64)
    psi = layer.init_state(2, "cpu")
    p_joint, _ = layer.probabilities(x, psi)
    torch.testing.assert_close(p_joint[..., 0], torch.ones(2, 1, dtype=torch.float64))
    p_joint.sum().backward()
    for name, param in layer.named_parameters():
        assert torch.isfinite(param.grad).all(), name


def test_complex64_default_and_double_promotion():
    layer = SQSLayer(3, 2, 1, 1)
    assert layer.weight.dtype is torch.float32 and layer.cdtype is torch.complex64
    assert layer.init_state(2, "cpu").dtype is torch.complex64
    layer.double()
    assert layer.cdtype is torch.complex128
    assert layer.init_state(2, "cpu").dtype is torch.complex128
    x = torch.rand(2, 1, 3, dtype=torch.float64)
    spikes, psi, aux = layer(x, layer.init_state(2, "cpu"))
    assert psi.dtype is torch.complex128 and spikes.dtype is torch.float64


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        SQSLayer(3, 2, 1, 1, ansatz="strongly_entangling")
    with pytest.raises(ValueError):
        SQSLayer(3, 2, 1, 1, surrogate="arctan")
    with pytest.raises(ValueError):
        SQSLayer(3, 2, 0, 1)
    with pytest.raises(ValueError):
        SQSLayer(3, 2, 1, 1, dtype=torch.float32)
    layer = SQSLayer(3, 2, 2, 1)
    with pytest.raises(ValueError):
        layer.currents(torch.rand(2, 1, 3))  # wrong number of channels


def test_w_init_variants():
    torch.manual_seed(0)
    a = SQSLayer(1000, 8, w_init="normal")
    assert abs(a.weight.std().item() - 1 / math.sqrt(1000)) < 0.005
    b = SQSLayer(1000, 8, w_init="kaiming_uniform")
    assert b.weight.abs().max().item() <= 1 / math.sqrt(1000)
    c = SQSLayer(100, 8, w_init="normal", w_init_scale=0.5)
    assert abs(c.weight.std().item() - 0.5) < 0.05
    d = SQSLayer(10, 2000, theta_init="small")
    assert abs(d.theta_rx.std().item() - 0.1) < 0.01
    e = SQSLayer(10, 2000, theta_init="uniform")
    assert 0.0 <= e.theta_rx.min() and e.theta_rx.max() <= math.pi


@pytest.mark.parametrize("ansatz", ANSATZE)
def test_all_ansatze_train_end_to_end(ansatz):
    """Every ansatz must give finite, non-zero gradients on all its parameters."""
    torch.manual_seed(1)
    layer = SQSLayer(5, 3, 2, 2, ansatz=ansatz)
    x = torch.rand(4, 2, 5)
    psi = layer.init_state(4, "cpu")
    total = torch.zeros(())
    for _ in range(3):
        spikes, psi, aux = layer(x, psi)
        total = total + spikes.sum() + torch.log(aux["p_sel"].clamp_min(1e-8)).sum()
    total.backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_sampled_gradient_is_exact_for_the_papers_configuration():
    """The frozen-history estimator omits the score term d/dtheta log P(history).  For the
    paper's configuration that term is identically zero, so the estimator is EXACT.

    Scope: this audits ONE isolated neuron, a two-step teacher-forced scalar loss, and only
    `theta_crx`/`theta_rx`.  It omits the input weights, `theta_crx2`, the straight-through spike
    path, the second layer and the response-window CE that training actually uses.  It
    therefore shows only that the *recurrent collapse-history score term* vanishes for the
    paper's neuron -- NOT that the estimator is exact for training, which it is not once the
    spike path and a second layer are included.
    """

    def audit(ansatz, n_mem, theta_init, seed):
        torch.manual_seed(seed)
        layer = SQSLayer(3, 1, 1, n_mem, ansatz=ansatz, dtype=torch.complex128,
                         theta_init=theta_init)
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)
        xs = [torch.rand(1, 1, 3, dtype=torch.float64) for _ in range(2)]
        params = [layer.theta_crx, layer.theta_rx]

        def one(history):
            psi = layer.init_state(1, "cpu").to(torch.complex128)
            logp = 0.0
            for x, o in zip(xs, history):
                p_sel, psi = layer.forward_with_outcome(x, psi, torch.tensor([[o]]))
                logp = logp + torch.log(p_sel.clamp_min(1e-12)).squeeze()
            p_final, _ = layer.probabilities(xs[-1], psi)
            return -torch.log(p_final[0, 0, 0].clamp_min(1e-12)), logp

        layer.zero_grad(set_to_none=True)
        total = 0.0
        for h in itertools.product((0, 1), repeat=2):     # exact: P(h) differentiated too
            loss, logp = one(h)
            total = total + torch.exp(logp) * loss
        total.backward()
        exact = torch.cat([p.grad.reshape(-1).clone() for p in params])

        approx = torch.zeros_like(exact)
        for h in itertools.product((0, 1), repeat=2):     # the trainer's scheme: history frozen
            layer.zero_grad(set_to_none=True)
            loss, logp = one(h)
            weight = torch.exp(logp).detach()
            loss.backward()
            approx = approx + weight * torch.cat([p.grad.reshape(-1).clone() for p in params])
        return exact, approx

    for seed in range(3):
        exact, approx = audit("crx_all", 1, "small", seed)      # the paper's configuration
        assert torch.allclose(exact, approx, atol=1e-12), (seed, (exact - approx).norm())

        exact, approx = audit("crx_both", 1, "uniform", seed)   # real temporal structure
        score = (exact - approx).norm() / exact.norm()
        cos = torch.nn.functional.cosine_similarity(exact, approx, dim=0)
        assert score > 1e-3, (seed, score)          # the score term is genuinely non-zero here
        assert cos > 0.8, (seed, cos)               # but the direction is not reversed
