"""``psi_pre`` and ``p_joint`` must match the same circuit built from PennyLane's gates.

The reference circuit is built exactly as described in the paper (Fig. 3 and its
generalisation): ``RX(phi_n)`` on IO wire ``n``, then the CRX layer(s), then ``RX`` on every
wire. PennyLane's ``qml.matrix`` with ``wire_order=range(N + N_mem)`` is big-endian with wire 0
the most significant bit, which is exactly our statevector layout, so no re-indexing is needed.

CRX wiring per ansatz (``wires=[control, target]``):
  ``crx_all``      IO -> memory, ``wires=[n, N+m]``
  ``crx_both``     the same layer, then memory -> IO, ``wires=[N+m, n]``, using ``theta_crx2``

Skipped when PennyLane is not installed (it is in the ``dev`` dependency group).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sqs_bench.neurons.sqs import SQSLayer

qml = pytest.importorskip("pennylane")

# every ansatz on the paper's (1,1) config and on the largest register we test
ANSATZ_CASES = [
    ("crx_all", 1, 1),
    ("crx_all", 2, 1),
    ("crx_all", 2, 2),
    ("crx_both", 1, 1),
    ("crx_both", 2, 2),
]
SEED_OFFSET = {"crx_all": 0, "crx_both": 14}


def _reference(n_io, n_mem, state, phi, theta_crx, theta_rx, ansatz, theta_crx2=None):
    """Return (statevector, IO probabilities) from PennyLane's gate matrices."""
    n_tot = n_io + n_mem

    def _crx_layer(theta, io_controls):
        for n in range(n_io):
            for m in range(n_mem):
                wires = [n, n_io + m] if io_controls else [n_io + m, n]
                qml.CRX(theta[n, m], wires=wires)

    def _ops():
        for n in range(n_io):
            qml.RX(phi[n], wires=n)
        if ansatz == "crx_all":
            _crx_layer(theta_crx, True)
        elif ansatz == "crx_both":
            _crx_layer(theta_crx, True)
            _crx_layer(theta_crx2, False)
        else:
            raise AssertionError(ansatz)
        for q in range(n_tot):
            qml.RX(theta_rx[q], wires=q)

    unitary = np.asarray(qml.matrix(_ops, wire_order=list(range(n_tot)))())
    out = unitary @ state
    probs = (np.abs(out) ** 2).reshape(2**n_io, 2**n_mem).sum(-1)
    return out, probs


def _compare_with_pennylane(layer, x, psi0):
    n_io, n_mem = layer.N, layer.N_mem
    b, k = psi0.shape[:2]
    p_joint, psi_pre = layer.probabilities(x, psi0)
    assert psi_pre.shape == (b, k, 2 ** (n_io + n_mem))
    assert p_joint.shape == (b, k, 2**n_io)

    phi = layer.angles(layer.currents(x)).detach().numpy()
    tc = layer.theta_crx.detach().numpy()
    tc2 = None if layer.theta_crx2 is None else layer.theta_crx2.detach().numpy()
    tr = layer.theta_rx.detach().numpy()
    for bi in range(b):
        for ki in range(k):
            ref_state, ref_probs = _reference(
                n_io,
                n_mem,
                psi0[bi, ki].numpy(),
                phi[bi, ki],
                tc[ki],
                tr[ki],
                layer.ansatz,
                None if tc2 is None else tc2[ki],
            )
            np.testing.assert_allclose(
                psi_pre[bi, ki].detach().numpy(), ref_state, atol=1e-6, rtol=0
            )
            np.testing.assert_allclose(
                p_joint[bi, ki].detach().numpy(), ref_probs, atol=1e-6, rtol=0
            )


@pytest.mark.parametrize("ansatz,n_io,n_mem", ANSATZ_CASES)
def test_psi_pre_matches_pennylane(ansatz, n_io, n_mem):
    torch.manual_seed(101 + 10 * n_io + n_mem + SEED_OFFSET[ansatz])
    b, k, p = 2, 3, 5
    layer = SQSLayer(p, k, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128)
    assert layer.n_in_channels == n_io and layer.n_out_channels == n_io

    x = torch.rand(b, n_io, p, dtype=torch.float64)
    # a *random* joint state, i.e. the memory qubits are entangled with whatever
    # they were entangled with before; this exercises the gates, not the reset.
    dim = 2 ** (n_io + n_mem)
    psi0 = torch.randn(b, k, dim, dtype=torch.complex128)
    psi0 = psi0 / psi0.norm(dim=-1, keepdim=True)
    _compare_with_pennylane(layer, x, psi0)


@pytest.mark.parametrize("ansatz,n_io,n_mem", ANSATZ_CASES)
def test_ground_state_start_matches_pennylane(ansatz, n_io, n_mem):
    """Same check from the |0...0> initial state used at t = 1."""
    torch.manual_seed(7)
    b, k, p = 2, 2, 4
    layer = SQSLayer(p, k, n_io, n_mem, ansatz=ansatz, dtype=torch.complex128)
    x = torch.rand(b, n_io, p, dtype=torch.float64)
    _compare_with_pennylane(layer, x, layer.init_state(b, "cpu"))
