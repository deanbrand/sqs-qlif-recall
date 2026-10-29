"""Post-measurement state: paper Eqs. (15)-(16).

For a joint IO outcome ``j`` the memory register must collapse to
``(<j| (x) I)|psi_pre> / sqrt(p_j)`` and the IO qubits must be reset to
``|0...0>``, i.e. ``psi_next = |0...0>_IO (x) mem_j``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sqs_bench.neurons.sqs import ANSATZE, SQSLayer

CASES = [(1, 1), (2, 1), (2, 2), (1, 2)]


@pytest.mark.parametrize("ansatz", ANSATZE)
@pytest.mark.parametrize("n_io,n_mem", CASES)
def test_collapse_matches_projection(n_io, n_mem, ansatz):
    torch.manual_seed(11 + n_io * 3 + n_mem)
    b, k, p = 3, 2, 4
    dim_io, dim_mem = 2**n_io, 2**n_mem
    layer = SQSLayer(p, k, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128)
    x = torch.rand(b, n_io, p, dtype=torch.float64)
    psi0 = torch.randn(b, k, dim_io * dim_mem, dtype=torch.complex128)
    psi0 = psi0 / psi0.norm(dim=-1, keepdim=True)

    p_joint, psi_pre = layer.probabilities(x, psi0)
    pre = psi_pre.detach().numpy().reshape(b, k, dim_io, dim_mem)
    pj = p_joint.detach().numpy()

    for j in range(dim_io):
        outcome = torch.full((b, k), j, dtype=torch.int64)
        p_sel, psi_next = layer.forward_with_outcome(x, psi0, outcome)
        np.testing.assert_allclose(p_sel.detach().numpy(), pj[:, :, j], atol=1e-12)

        nxt = psi_next.detach().numpy()
        # IO qubits are back in |0...0>: only the first 2**N_mem entries survive.
        np.testing.assert_allclose(nxt[..., dim_mem:], 0.0, atol=1e-14)
        expected = pre[:, :, j, :] / np.sqrt(pj[:, :, j])[..., None]
        np.testing.assert_allclose(nxt[..., :dim_mem], expected, atol=1e-10)
        # unit norm
        np.testing.assert_allclose((np.abs(nxt) ** 2).sum(-1), 1.0, atol=1e-10)


@pytest.mark.parametrize("ansatz", ANSATZE)
@pytest.mark.parametrize("n_io,n_mem", CASES)
def test_probabilities_sum_to_one_and_marginals_consistent(n_io, n_mem, ansatz):
    torch.manual_seed(3)
    b, k, p = 4, 3, 6
    layer = SQSLayer(p, k, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128)
    x = torch.rand(b, n_io, p, dtype=torch.float64)
    psi = layer.init_state(b, "cpu")
    spikes, psi_next, aux = layer(x, psi)

    np.testing.assert_allclose(aux["p_joint"].sum(-1).detach().numpy(), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        (psi_next.abs() ** 2).sum(-1).detach().numpy(), 1.0, atol=1e-10
    )
    # marginal P(s_n = 1) recomputed by hand from the joint
    pj = aux["p_joint"].detach().numpy()
    bits = np.array(
        [[(j >> (n_io - 1 - n)) & 1 for n in range(n_io)] for j in range(2**n_io)],
        dtype=float,
    )
    np.testing.assert_allclose(aux["p"].detach().numpy(), pj @ bits, atol=1e-12)
    assert spikes.shape == (b, k, n_io)
    assert set(np.unique(spikes.detach().numpy())) <= {0.0, 1.0}


@pytest.mark.parametrize("ansatz", ANSATZE)
@pytest.mark.parametrize("n_io,n_mem", CASES)
def test_forward_with_outcome_agrees_with_forward(n_io, n_mem, ansatz):
    torch.manual_seed(99)
    b, k, p = 5, 4, 4
    layer = SQSLayer(p, k, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128)
    x = torch.rand(b, n_io, p, dtype=torch.float64)
    psi = layer.init_state(b, "cpu")
    _, psi_next, aux = layer(x, psi)
    p_sel, psi_next_forced = layer.forward_with_outcome(x, psi, aux["outcome"])
    torch.testing.assert_close(p_sel, aux["p_sel"])
    torch.testing.assert_close(psi_next_forced, psi_next)


@pytest.mark.parametrize("n_io,n_mem", CASES)
def test_spike_bits_are_the_outcome_bits(n_io, n_mem):
    torch.manual_seed(5)
    b, k, p = 6, 3, 4
    layer = SQSLayer(p, k, n_io, n_mem)
    x = torch.rand(b, n_io, p)
    psi = layer.init_state(b, "cpu")
    spikes, _, aux = layer(x, psi)
    j = aux["outcome"].numpy()
    bits = np.stack([(j >> (n_io - 1 - n)) & 1 for n in range(n_io)], axis=-1)
    np.testing.assert_array_equal(spikes.detach().numpy(), bits.astype(np.float32))
    # p_sel is the probability of the realised outcome
    pj = aux["p_joint"].detach().numpy()
    np.testing.assert_allclose(
        aux["p_sel"].detach().numpy(), np.take_along_axis(pj, j[..., None], -1)[..., 0], atol=1e-7
    )
