"""Sampling, memory and readout of the SQS neuron, checked against exact calculations.

* The sampler must reproduce the exact spike-trajectory distribution. For one neuron with
  N = N_mem = 1 and T = 3 there are 8 spike trajectories. Their exact probabilities are
  obtained here in numpy by recursive projection -- building the 4x4 unitary by hand from
  ``RX``/``CRX`` matrices -- and compared with 100k single-shot trajectories drawn by
  ``SQSLayer``.
* The published circuit (``crx_all``) at N_mem = 1 cannot read its memory: no Born probability
  depends on the memory state, to machine precision, at either initialisation. The read gate
  (``crx_both``) removes that, also at its small initialisation.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from sqs_bench.neurons.sqs import SQSLayer

T = 3
N_DRAWS = 100_000


def _rx(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _crx(a):
    m = np.eye(4, dtype=np.complex128)
    m[2:, 2:] = _rx(a)
    return m


def _unitary(phi, theta_crx, theta_rx):
    """U = (RX(t2) (x) RX(t3)) . CRX(t1) . (RX(phi) (x) I), IO qubit first."""
    embed = np.kron(_rx(phi), np.eye(2, dtype=np.complex128))
    single = np.kron(_rx(theta_rx[0]), _rx(theta_rx[1]))
    return single @ _crx(theta_crx) @ embed


def _exact_trajectory_probs(phis, theta_crx, theta_rx):
    """Recursive projection over all 2**T trajectories; returns a dict."""
    out = {}

    def walk(t, mem, prob, traj):
        if t == T:
            out[tuple(traj)] = prob
            return
        psi = _unitary(phis[t], theta_crx, theta_rx) @ np.concatenate([mem, [0.0, 0.0]])
        for s in (0, 1):
            block = psi[2 * s : 2 * s + 2]
            p = float(np.vdot(block, block).real)
            if p <= 0.0:
                out_traj = traj + [s]
                walk(t + 1, np.zeros(2, dtype=np.complex128), 0.0, out_traj)
                continue
            walk(t + 1, block / np.sqrt(p), prob * p, traj + [s])

    walk(0, np.array([1.0, 0.0], dtype=np.complex128), 1.0, [])
    return out


def test_trajectory_distribution_matches_exact():
    chisquare = pytest.importorskip("scipy.stats").chisquare
    torch.manual_seed(1234)
    np.random.seed(0)
    p_in = 4
    layer = SQSLayer(p_in, 1, 1, 1, dtype=torch.complex128)
    # fixed inputs, identical for every sample in the batch
    xs = [torch.rand(1, 1, p_in, dtype=torch.float64) for _ in range(T)]

    with torch.no_grad():
        phis = [float(layer.angles(layer.currents(x))[0, 0, 0]) for x in xs]
    theta_crx_t = layer.theta_crx.detach()
    theta_crx = float(theta_crx_t[0, 0, 0])
    theta_rx = layer.theta_rx[0].detach().numpy()
    assert max(phis) > 0.0, "degenerate test: every input current was <= 0"
    exact = _exact_trajectory_probs(phis, theta_crx, theta_rx)
    keys = list(itertools.product((0, 1), repeat=T))
    p_exact = np.array([exact[k] for k in keys])
    assert abs(p_exact.sum() - 1.0) < 1e-10

    # sample N_DRAWS independent trajectories in one batch
    x_batch = [x.expand(N_DRAWS, 1, p_in).contiguous() for x in xs]
    psi = layer.init_state(N_DRAWS, "cpu")
    bits = []
    with torch.no_grad():
        for t in range(T):
            spikes, psi, _ = layer(x_batch[t], psi)
            bits.append(spikes[:, 0, 0])
    codes = sum(int(2 ** (T - 1 - t)) * bits[t].to(torch.int64) for t in range(T))
    counts = torch.bincount(codes, minlength=2**T).numpy()

    p_emp = counts / N_DRAWS
    tv = 0.5 * np.abs(p_emp - p_exact).sum()
    assert tv < 0.01, f"TV distance {tv:.4f}\nexact={p_exact}\nempirical={p_emp}"

    keep = p_exact * N_DRAWS > 5  # chi-square validity
    stat, pval = chisquare(counts[keep], p_exact[keep] * N_DRAWS)
    assert pval > 0.001, f"chi-square p = {pval:.5f} (stat {stat:.2f}), tv = {tv:.4f}"


def test_first_step_marginal_matches_born_rule():
    """Sanity check of the sampler on a single step (no memory involved yet)."""
    torch.manual_seed(2)
    layer = SQSLayer(3, 2, 1, 1, dtype=torch.complex128)
    x = torch.rand(1, 1, 3, dtype=torch.float64).expand(N_DRAWS, 1, 3).contiguous()
    psi = layer.init_state(N_DRAWS, "cpu")
    with torch.no_grad():
        spikes, _, aux = layer(x, psi)
    p = aux["p"][0, :, 0].numpy()
    rate = spikes[:, :, 0].mean(0).numpy()
    assert np.all(np.abs(rate - p) < 4.0 * np.sqrt(p * (1 - p) / N_DRAWS) + 1e-3)


def test_memory_actually_carries_information():
    """The state after a step must depend on the measured outcome (otherwise the
    'memory' qubit would be useless)."""
    torch.manual_seed(4)
    layer = SQSLayer(3, 1, 1, 1, dtype=torch.complex128)
    x = torch.rand(1, 1, 3, dtype=torch.float64)
    psi = layer.init_state(1, "cpu")
    _, psi0 = layer.forward_with_outcome(x, psi, torch.zeros(1, 1, dtype=torch.int64))
    _, psi1 = layer.forward_with_outcome(x, psi, torch.ones(1, 1, dtype=torch.int64))
    assert not torch.allclose(psi0, psi1, atol=1e-6)


def _memory_probability_spread(n_io, n_mem, ansatz, theta_init="uniform", trials=10, states=6):
    """Max spread of the Born probabilities over random memory states."""
    spread = 0.0
    for _ in range(trials):
        layer = SQSLayer(
            3, 1, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128, theta_init=theta_init
        )
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)  # drive the RX embedding
        x = torch.rand(1, n_io, 3, dtype=torch.float64)
        dim, dim_mem = 2 ** (n_io + n_mem), 2**n_mem
        probs = []
        for _ in range(states):
            m = torch.randn(dim_mem, dtype=torch.complex128)
            m = m / m.norm()
            psi = torch.zeros(1, 1, dim, dtype=torch.complex128)
            psi[0, 0, :dim_mem] = m  # IO in |0...0>, memory in a random state
            p_joint, _ = layer.probabilities(x, psi)
            probs.append(p_joint.detach().reshape(-1))
        stacked = torch.stack(probs)
        spread = max(spread, (stacked.max(0).values - stacked.min(0).values).max().item())
    return spread


def test_memory_state_does_not_affect_output_when_nmem_is_1():
    """Characterisation of the paper's shallow ansatz, not a bug in this code.

    With ``ansatz="crx_all"`` every CRX is *controlled by an IO qubit and targets
    a memory qubit*, and the IO register enters each step in ``|0...0>``, so the
    state before the ansatz is a product ``|chi> (x) |m>``.  After the CRX layer
    the state is ``sum_a c_a |a> (x) R_a|m>`` with ``R_a`` a product of memory
    RX rotations; for ``N_mem = 1`` all the ``R_a`` are rotations about the same
    axis, so ``R_a^dag R_b = RX(beta_b - beta_a)`` and the overlap is
    ``<m|RX(D)|m> = cos(D/2) - i sin(D/2) <X>_m``.  The memory-dependent part is
    purely *imaginary*, and the Born probabilities depend only on the real part,
    so they are memory-blind.  (The overlap itself is NOT simply ``cos(D/2)`` --
    an X eigenstate is an immediate counterexample.)  Equivalently: the two
    memory Kraus operators satisfy ``K_b^dag K_b ~ I``, so the normalised
    collapse is *unitary* on the memory -- see
    ``test_collapse_is_unitary_on_the_memory_so_nothing_is_overwritten``.

    Consequence: for the paper's configuration ``N = N_mem = 1`` the SQS neuron's
    *spike statistics* carry no memory - the spikes are conditionally independent
    across time given the inputs.  Two or more memory qubits break the degeneracy.
    """
    torch.manual_seed(3)
    for n_io, n_mem, memoryless in [(1, 1, True), (2, 1, True), (1, 2, False), (2, 2, False)]:
        spread = _memory_probability_spread(n_io, n_mem, "crx_all")
        if memoryless:
            assert spread < 1e-12, (n_io, n_mem, spread)
        else:
            assert spread > 1e-2, (n_io, n_mem, spread)


@pytest.mark.parametrize("theta_init", ["small", "uniform"])
def test_crx_all_is_memory_blind_at_both_initialisations(theta_init):
    """The blindness is a property of the wiring, not of the angles: ``sqsnn`` (small init)
    and ``sqsnn_uni`` (uniform init) are both exactly blind at N_mem = 1."""
    torch.manual_seed(3)
    for n_io in (1, 2):
        spread = _memory_probability_spread(n_io, 1, "crx_all", theta_init=theta_init)
        assert spread < 1e-12, (theta_init, n_io, spread)


@pytest.mark.parametrize("theta_init", ["small", "uniform"])
@pytest.mark.parametrize("n_io,n_mem", [(1, 1), (2, 1), (2, 2)])
def test_memory_state_does_affect_output_when_memory_controls_io(n_io, n_mem, theta_init):
    """Wiring a CRX from the memory register to an IO qubit (the read gate of
    ``crx_both``) removes the degeneracy above: the Born probability varies with the
    memory state even at ``N_mem = 1``, at the small initialisation of ``sqsnn_read``
    as well as at the uniform one of ``sqsnn_read_uni``."""
    torch.manual_seed(21)
    spread = _memory_probability_spread(n_io, n_mem, "crx_both", theta_init=theta_init)
    assert spread > 1e-2, f"crx_both ({n_io},{n_mem}, {theta_init}) still memory-blind: {spread:g}"


def _grad_reach(n_mem, ansatz, source_step, T=5, seed=0):
    """max |d(loss at the final step) / d(input at ``source_step``)|.

    Non-zero means BPTT can carry credit from that step to a later output, i.e. a memory there
    could in principle be *learned*.  Zero means there is no path at all.
    """
    torch.manual_seed(seed)
    layer = SQSLayer(4, 2, 1, n_mem, ansatz=ansatz, dtype=torch.complex128)
    with torch.no_grad():
        layer.weight.abs_().add_(0.3)
    xs = [torch.rand(8, 1, 4, dtype=torch.float64, requires_grad=(t == source_step))
          for t in range(T)]
    psi = layer.init_state(8, "cpu").to(torch.complex128)
    for t in range(T):
        _, psi, aux = layer.forward(xs[t], psi)
    aux["p"].sum().backward()
    return xs[source_step].grad.abs().max().item()


def test_no_credit_path_across_time_in_the_papers_configuration():
    """Stronger than the Born-probability degeneracy: at N=N_mem=1 with ``crx_all`` there is no
    gradient path from ANY past input to a later output.

    This separates two explanations of a chance-level recall result that would otherwise be
    confounded: "the neuron has no memory" versus "the memory exists but BPTT cannot train it
    through the sampling and collapse".  Here it is the former; at N_mem=2 credit does flow.
    """
    for source in range(4):
        assert _grad_reach(1, "crx_all", source) < 1e-12, source
        assert _grad_reach(2, "crx_all", source) > 1e-3, source


def test_collapse_is_unitary_on_the_memory_so_nothing_is_overwritten():
    """The conditional collapse PRESERVES the memory, it does not erase it.

    At N=N_mem=1 the two memory Kraus operators satisfy K_b^dag K_b proportional to the identity,
    so after ``_collapse`` divides by the branch probability each outcome acts as a *unitary* on
    the memory: overlaps, and hence distinguishability, are preserved exactly.

    This matters because it separates two very different diagnoses of the paper's circuit:
    "the state is overwritten every step" (it is not) versus "the state is retained but no
    output can ever depend on it" (it is). Only the second is supported.
    """
    torch.manual_seed(0)
    layer = SQSLayer(4, 1, 1, 1, ansatz="crx_all", dtype=torch.complex128)
    with torch.no_grad():
        layer.weight.abs_().add_(0.3)
    x = torch.rand(1, 1, 4, dtype=torch.float64)

    def memory_after(mem_vec, outcome):
        psi = torch.zeros(1, 1, 4, dtype=torch.complex128)
        psi[0, 0, :2] = mem_vec
        _, psi_next = layer.forward_with_outcome(x, psi, torch.tensor([[outcome]]))
        return psi_next[0, 0, :2]

    for trial in range(5):
        a = torch.randn(2, dtype=torch.complex128); a = a / a.norm()
        b = torch.randn(2, dtype=torch.complex128); b = b / b.norm()
        before = torch.vdot(a, b).abs()
        for outcome in (0, 1):
            after = torch.vdot(memory_after(a, outcome), memory_after(b, outcome)).abs()
            assert torch.allclose(before, after, atol=1e-12), (trial, outcome, before, after)


def _readout_and_write(n_mem, ansatz, theta_init, trials=9):
    """(readout sensitivity, step-0 write strength), each the median over ``trials`` layers.

    A circuit can only hold a cue and report it if BOTH are non-negligible: the memory must be
    able to influence the output, AND the step-0 input must be able to reach the memory.
    """
    reads, writes = [], []
    for t in range(trials):
        torch.manual_seed(200 + t)
        layer = SQSLayer(4, 1, 1, n_mem, ansatz=ansatz, dtype=torch.complex128,
                         theta_init=theta_init)
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)
        x = torch.rand(1, 1, 4, dtype=torch.float64)
        probs = []
        for _ in range(6):
            m = torch.randn(2 ** n_mem, dtype=torch.complex128); m = m / m.norm()
            psi = torch.zeros(1, 1, 2 ** (1 + n_mem), dtype=torch.complex128)
            psi[0, 0, :2 ** n_mem] = m
            p, _ = layer.probabilities(x, psi)
            probs.append(p.detach().reshape(-1))
        st = torch.stack(probs)
        reads.append((st.max(0).values - st.min(0).values).max().item())
        xb = torch.rand(6, 1, 4, dtype=torch.float64)
        _, psi1, _ = layer.forward(xb, layer.init_state(6, "cpu").to(torch.complex128))
        pops = (psi1.reshape(6, 1, 2, 2 ** n_mem).abs() ** 2).sum(2)[:, 0]
        writes.append(pops.std(0).max().item())
    reads.sort(); writes.sort()
    return reads[len(reads) // 2], writes[len(writes) // 2]


def test_readout_and_write_screening():
    """Readability AND step-0 writability, measured without any training.

    The published circuit writes its memory but cannot read it; the read gate can do both.
    """
    for theta_init in ("small", "uniform"):
        read, write = _readout_and_write(1, "crx_all", theta_init)   # the paper's configuration
        assert read < 1e-12, (theta_init, read)                      # unreadable, by the theorem
        assert write > 1e-4, (theta_init, write)                     # ... although it is written

    read, _ = _readout_and_write(1, "crx_both", "small")             # sqsnn_read: weak, not zero
    assert read > 1e-3, read

    for n_mem, ansatz in [(1, "crx_both"), (2, "crx_all")]:          # readable and writable
        read, write = _readout_and_write(n_mem, ansatz, "uniform")
        assert read > 1e-3, (ansatz, n_mem, read)
        assert write > 1e-3, (ansatz, n_mem, write)


def test_cue_information_is_retained_undiminished_across_distractors():
    """Within a given outcome history the cue separation does not decay.

    The collapse is unitary on the memory branch by branch, so once a history is fixed the four
    cue classes keep their pairwise trace distance as distractors accumulate, to 12 decimal
    places.

    Note: the *magnitude* of that distance is a property of the post-selected branch, not of the
    memory.  Forcing outcome 1 everywhere instead of 0 gives a very different value, and the
    branch measured here can be nearly unreachable for some cue classes.  This test pins the
    *absence of decay within a branch* only; the Born-weighted ensemble is a different and
    smaller quantity.
    """
    K, D = 4, 5

    def memory_states(n_distractors):
        torch.manual_seed(0)
        layer = SQSLayer(D, 1, 1, 1, ansatz="crx_all", dtype=torch.complex128, theta_init="small")
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)
        g = torch.Generator().manual_seed(7)
        noise = (torch.rand(n_distractors, 1, D, generator=g) < 0.1).to(torch.float64)
        states = []
        for y in range(K):
            cue = torch.zeros(1, 1, D, dtype=torch.float64)
            cue[0, 0, y] = 1.0
            psi = layer.init_state(1, "cpu").to(torch.complex128)
            for x in [cue] + list(noise.unsqueeze(1)):
                _, psi = layer.forward_with_outcome(
                    x.reshape(1, 1, D), psi, torch.zeros(1, 1, dtype=torch.int64))
            v = psi[0, 0, :2]
            states.append(v / v.norm())
        return states

    def mean_trace_distance(states):
        ds = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                ov = torch.vdot(states[i], states[j]).abs() ** 2
                ds.append(torch.sqrt(torch.clamp(1 - ov, min=0)))
        return torch.stack(ds).mean()

    base = mean_trace_distance(memory_states(0))
    assert base > 0.05, base                      # the cue IS written into the memory
    for k in (1, 2, 3, 4):                        # and distractors do not degrade it at all
        assert torch.allclose(mean_trace_distance(memory_states(k)), base, atol=1e-12), k


def _cue_to_output_signal(n_mem, ansatz, theta_init, lag, trials=7, seed0=300):
    """Max spread of the OUTPUT probabilities across cue classes, `lag` steps after the cue.

    Retention is not enough: to be learnable the cue must reach an *observable*.  This is the
    recall task in miniature -- four one-hot cues on D = 5 lines, ``lag`` noise steps, then a go
    pulse -- and it is measurable without any training.
    """
    K, D = 4, 5
    vals = []
    for t in range(trials):
        torch.manual_seed(seed0 + t)
        layer = SQSLayer(D, 1, 1, n_mem, ansatz=ansatz, dtype=torch.complex128,
                         theta_init=theta_init)
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)
        g = torch.Generator().manual_seed(11)
        noise = (torch.rand(max(lag, 1), 1, D, generator=g) < 0.1).to(torch.float64)
        probe = torch.zeros(1, 1, D, dtype=torch.float64)
        probe[0, 0, K] = 1.0                       # the go pulse
        zero = torch.zeros(1, 1, dtype=torch.int64)
        states = []
        for y in range(K):
            cue = torch.zeros(1, 1, D, dtype=torch.float64)
            cue[0, 0, y] = 1.0
            _, psi = layer.forward_with_outcome(cue, layer.init_state(1, "cpu").to(torch.complex128), zero)
            for k in range(lag):
                _, psi = layer.forward_with_outcome(noise[k].reshape(1, 1, D), psi, zero)
            states.append(psi)
        st = torch.stack([layer.probabilities(probe, s)[0].detach().reshape(-1) for s in states])
        vals.append((st.max(0).values - st.min(0).values).max().item())
    vals.sort()
    return vals[len(vals) // 2]


def test_cue_reaches_the_output_only_when_the_memory_can_be_read():
    """The paper's ansatz leaks no cue information into any observable, at any lag.

    ``crx_both`` -- one extra CRX layer, so the memory conditions the IO register -- does.  At
    the small initialisation of ``sqsnn_read`` the signal is weak (about 1e-4) but more than ten
    orders of magnitude above ``crx_all``.  Retention was never the issue: see
    test_cue_information_is_retained_undiminished_across_distractors.
    """
    for lag in (0, 1, 2):
        assert _cue_to_output_signal(1, "crx_all", "small", lag) < 1e-12, lag
        assert _cue_to_output_signal(1, "crx_all", "uniform", lag) < 1e-12, lag
        assert _cue_to_output_signal(1, "crx_both", "uniform", lag) > 1e-3, lag
        assert _cue_to_output_signal(1, "crx_both", "small", lag) > 1e-5, lag
