r"""Stochastic quantum spiking (SQS) neuron -- batched, differentiable statevector simulator.

Reference
---------
J. Chen, B. Rajendran and O. Simeone, *Stochastic Quantum Spiking Neural Networks with
Quantum Memory and Local Learning*, IEEE J. Sel. Areas Commun. (2026), arXiv:2506.21324,
Sec. III (neuron model) and Appendix A (surrogate gradient). "The paper" below means this one.

Model (one neuron, one time step ``t``)
---------------------------------------
A neuron owns ``N`` input-output (IO) qubits and ``N_mem`` memory qubits,
``N_tot = N + N_mem``.  At the start of every step the IO qubits are in
:math:`|0\rangle^{\otimes N}` while the memory qubits carry the (normalised,
conditional) state left behind by the previous measurement -- paper Eq. (16):

.. math::
    \rho^{\rm IM}_{t+1}
      = U(\mathbf z_t,\boldsymbol\theta)\,
        \bigl(|0\rangle\!\langle 0|^{\otimes N}\otimes\sigma^{\rm mem}_t\bigr)\,
        U(\mathbf z_t,\boldsymbol\theta)^\dagger .

The step is

1. **Input currents** (paper Eq. (10), no bias)

   .. math:: z_t^n = \mathbf w^n \mathbf x_t^n ,\qquad n = 1,\dots,N .

2. **Quantum embedding** (paper Eq. (11) with :math:`f_s(z)=\pi z`)

   .. math:: \phi_t^n = \pi z_t^n \ \text{if}\ z_t^n>0,\quad 0\ \text{otherwise},

   followed by :math:`R_X(\phi_t^n)` on IO qubit ``n``, with
   :math:`R_X(a)=\exp(-i a X/2)=\begin{pmatrix}\cos\frac a2 & -i\sin\frac a2\\
   -i\sin\frac a2 & \cos\frac a2\end{pmatrix}`.

3. **Quantum memory retrieval** (paper Eq. (12)): the ansatz :math:`V(\theta)`.
   Two wirings are available.  Both iterate the CRX gates row-major over
   ``(n, m)`` and end with an ``RX`` on every qubit (IO first, then memory):

   * ``"crx_all"`` (default): the published SQS circuit.  ``CRX(theta_{n,m})``
     with **control = IO qubit n, target = memory qubit m**.  For
     ``N = N_mem = 1`` this is exactly the paper's Fig. 3 circuit:
     ``CRX(theta_1)``, ``RX(theta_2)`` on the IO qubit, ``RX(theta_3)`` on the
     memory qubit.  ``G = N*N_mem + N + N_mem`` (3 for (1,1), 5 for (2,1), 8 for
     (2,2)).
   * ``"crx_both"``: the published circuit plus a **read gate**.  The IO -> memory
     layer (``theta_crx``) is followed by a memory -> IO layer (``theta_crx2``,
     control = memory qubit m, target = IO qubit n), so the memory controls the
     IO qubits, then the ``RX`` layer.  ``G = 2*N*N_mem + N + N_mem``.

   ``G`` is returned by :func:`pqc_param_count`; the angles are per neuron.

4. **Single-shot measurement** (paper Eqs. (14)-(15)): the *joint* outcome
   :math:`\mathbf s_t\in\{0,1\}^N` of all IO qubits is drawn once from the Born
   marginal :math:`p(\mathbf s_t)=\langle\mathbf s_t|\rho^{\rm I}_t|\mathbf s_t\rangle`,
   and the memory register collapses to
   :math:`(\langle\mathbf s_t|\otimes I)|\psi\rangle / \sqrt{p(\mathbf s_t)}`.
   The output spikes are the bits of :math:`\mathbf s_t`.  The draw uses the
   global torch RNG of the compute device.

Qubit ordering
--------------
The statevector is ``[B, K, 2**N_tot]``, **big-endian with the IO qubits first**:
qubit ``q = 0`` is the most significant bit and basis index
``j = sum_q bit_q * 2**(N_tot-1-q)``.  Qubits ``0..N-1`` are IO,
``N..N_tot-1`` are memory, so ``j = j_io * 2**N_mem + j_mem``.  This is
PennyLane's default wire ordering, hence ``psi_pre`` can be compared directly
against ``qml.state()`` for the circuit
``qml.RX(phi_n, wires=n)``, ``qml.CRX(theta, wires=[n, N+m])`` (and
``wires=[N+m, n]`` for the read gate of ``crx_both``), ``qml.RX(theta, wires=q)``.

Surrogate gradient (paper Appendix A)
-------------------------------------
The sampled spike :math:`s^n_t` is not differentiable.  Appendix A replaces it in
the backward pass with :math:`\sigma(p)`, the sigmoid of the *spiking
probability*, giving :math:`\partial s/\partial p=\sigma(p)(1-\sigma(p))`.  This
is implemented as a straight-through estimator, so the forward value stays the
hard 0/1 sample:

* ``surrogate="marginal_sigmoid"`` (default, faithful): ``p`` is the per-bit Born
  marginal ``P(s_n = 1)``; ``s = s_hard + (sigmoid(p) - sigmoid(p).detach())``.
* ``surrogate="joint_sigmoid"``: ``p`` is the probability of the *realised joint*
  outcome, ``p_sel``, reused for every bit.  This matches a literal reading of
  Appendix A (which writes :math:`p(\mathbf s_t|\cdot)` for the whole vector),
  but note the gradient is then the same for all ``N`` bits and, for a bit that
  came out 0, points in the direction of making that same joint outcome more
  likely rather than making the bit fire.
* ``surrogate="ste"``: plain identity straight-through, ``ds/dp = 1``.

For ``N = 1`` the marginal and the joint coincide up to which outcome was drawn
(``p_sel = p`` when ``s = 1`` and ``1 - p`` when ``s = 0``).

Gradients flow through the memory amplitudes across time steps (full BPTT
through the unitary chain).  The layer never detaches the state itself; the
caller truncates BPTT with :meth:`SpikingLayer.detach_state`.

The published circuit cannot read its memory
---------------------------------------------
With ``ansatz="crx_all"`` every CRX is controlled by an IO qubit and targets a
memory qubit, and the IO register enters each step in a computational *basis*
state -- ``|0...0>`` here, but a projective measurement of the IO qubits leaves
one whatever the reset convention, so the argument does not depend on that
choice.  The state before the ansatz is therefore always a *product*
``|chi> (x) |m>``, and after the CRX layer it is ``sum_a c_a |a> (x) R_a|m>``
with ``R_a`` a product of memory RX rotations.  For ``N_mem = 1`` all the
``R_a`` rotate about the same axis, so ``R_a^dag R_b = RX(D)`` and
``<m|RX(D)|m> = cos(D/2) - i sin(D/2) <X>_m``.  The memory-dependent term is
purely *imaginary* while the Born probabilities use only the real part:
**the Born probabilities do not depend on the memory state at all**.
(The overlap is NOT simply ``cos(D/2)``; an X eigenstate is a counterexample.)

The memory is *retained*, not erased: the two Kraus operators satisfy
``K_b^dag K_b ~ I``, so the normalised collapse is a **unitary** map on the
memory for each outcome and preserves overlaps exactly.  The defect is an
inaccessible readout channel, not an overwrite.  For the paper's own
configuration ``N = N_mem = 1`` the SQS neuron's spike statistics are thus
memoryless (conditionally independent across time given the inputs), even
though the statevector does evolve.  ``N_mem >= 2`` breaks the degeneracy, and
so does the read gate of ``crx_both``: there the memory controls a rotation *of*
the IO register, and the Born probabilities depend on ``|m>`` already at
``N_mem = 1``.  This is a property of the paper's model, not of this simulator.

Frozen read-gate angle
----------------------
``theta_crx2_scale`` and ``theta_crx2_trainable`` hold the architecture at
``ansatz="crx_both"`` -- same wiring, gate count and parameter tensors -- and vary
only the read-gate angles:

* ``theta_crx2_scale`` is the initialisation scale of the read gate
  (``theta_crx2``) **alone**.  ``None`` reuses ``theta_init_scale``.  Under
  ``theta_init="uniform"`` the angles are ``U(0, scale)``; because ``theta_crx2``
  is drawn last, the weights, the first CRX layer and the ``RX`` layer come off
  the same RNG draws at every scale, and at a fixed seed scale ``d`` gives
  exactly ``d * u`` with the same ``u``.  Two scales at one seed therefore differ
  in the read-gate angles and in nothing else.
* ``theta_crx2_trainable=False`` freezes those angles at their initial value.

At ``theta_crx2_scale=0.0`` every read-gate angle is exactly ``0``,
``CRX(0) = I``, and the circuit *is* ``crx_all``, so it cannot read its memory.

The read gate acts *before* the ``RX`` layer and the measurement, so it is not a
passive read: it moves the Born probabilities, hence the sampled outcome, the
collapsed state carried into the next step and every temporal gradient, in both
network layers.  A frozen arm optimises a restricted model, so it is expected to
score below a ``crx_both`` model that also trains ``theta_crx2``.

Complex-autograd caveats
------------------------
* Probabilities use ``psi.real**2 + psi.imag**2``.  Never ``psi.abs()**2``: the
  subgradient of ``abs`` at 0 is NaN and amplitudes *do* hit exactly 0 (the
  ``|0..0>`` reset and the zero-padding of the collapsed state).
* Autograd for complex tensors uses the conjugate Wirtinger derivative.  Because
  every leaf here is real and the loss is real, the accumulated ``.grad`` on the
  real parameters is the true gradient -- but do not mix in complex leaves.
* Complex64 and AMP do not mix; :meth:`forward` raises if autocast is enabled.
* The whole step is out-of-place (no index assignment into graph tensors), so it
  is ``torch.compile``- and ``vmap``-friendly, though compile is not required.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import SpikingLayer

__all__ = [
    "SQSLayer",
    "sqs_step_probabilities",
    "pqc_param_count",
    "apply_rx",
    "apply_crx",
]

_REAL_OF_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
_COMPLEX_OF_REAL: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

SURROGATES = ("marginal_sigmoid", "joint_sigmoid", "ste")
#: ``crx_all`` = the published SQS circuit; ``crx_both`` = adds the read gate
#: (a memory -> IO CRX layer), so the memory controls the IO qubits
ANSATZE = ("crx_all", "crx_both")
#: ansatze needing a second CRX layer (``theta_crx2``)
_TWO_LAYER_ANSATZE = ("crx_both",)


# --------------------------------------------------------------------------- #
# gate kernels (pure functions on [B, K, 2**n_tot] statevectors)
# --------------------------------------------------------------------------- #
def _shape_angle(angle: Tensor, n_extra: int) -> Tensor:
    """Reshape a ``[]`` / ``[K]`` / ``[B, K]`` angle so it broadcasts against a
    statevector view whose trailing ``n_extra`` axes are Hilbert-space axes."""
    if angle.dim() == 0:
        return angle
    if angle.dim() == 1:  # per-neuron angle [K] -> [1, K, 1, ...]
        return angle.reshape(1, -1, *([1] * n_extra))
    if angle.dim() == 2:  # per-(batch, neuron) angle [B, K] -> [B, K, 1, ...]
        return angle.reshape(*angle.shape, *([1] * n_extra))
    raise ValueError(f"angle must have 0, 1 or 2 dims, got {angle.dim()}")


def apply_rx(psi: Tensor, angle: Tensor, qubit: int, n_tot: int) -> Tensor:
    r"""Apply :math:`R_X(\text{angle})` to ``qubit`` of ``psi`` ``[B, K, 2**n_tot]``.

    ``angle`` may be ``[]``, ``[K]`` (per neuron) or ``[B, K]`` (per sample and
    neuron).  Out-of-place.
    """
    b, k, _ = psi.shape
    left = 1 << qubit
    right = 1 << (n_tot - 1 - qubit)
    v = psi.reshape(b, k, left, 2, right)
    a0 = v[:, :, :, 0, :]
    a1 = v[:, :, :, 1, :]
    half = 0.5 * angle
    cos = _shape_angle(torch.cos(half), 2)
    msin = _shape_angle(torch.sin(half), 2) * (-1j)  # -i sin(a/2)
    n0 = cos * a0 + msin * a1
    n1 = msin * a0 + cos * a1
    return torch.stack((n0, n1), dim=3).reshape(b, k, -1)


def apply_crx(psi: Tensor, angle: Tensor, control: int, target: int, n_tot: int) -> Tensor:
    r"""Apply ``CRX(angle) = |0><0| (x) I + |1><1| (x) RX(angle)`` -- the matrix
    written out in the paper's Sec. VI-A, i.e. ``RX`` on ``target`` conditioned on
    ``control`` being ``|1>``.

    ``control`` and ``target`` may be in either wire order (``crx_all`` controls
    on IO qubits, the read gate of ``crx_both`` on memory qubits).  Out-of-place.
    """
    if control == target:
        raise ValueError("apply_crx needs distinct control and target wires")
    b, k, _ = psi.shape
    lo, hi = (control, target) if control < target else (target, control)
    left = 1 << lo
    mid = 1 << (hi - lo - 1)
    right = 1 << (n_tot - 1 - hi)
    v = psi.reshape(b, k, left, 2, mid, 2, right)

    half = 0.5 * angle
    cos = _shape_angle(torch.cos(half), 3)
    msin = _shape_angle(torch.sin(half), 3) * (-1j)  # -i sin(a/2)

    if control < target:  # control on axis 3, target on axis 5
        ctrl0 = v[:, :, :, 0]  # [B, K, left, mid, 2, right] -- untouched
        ctrl1 = v[:, :, :, 1]
        a0, a1 = ctrl1[..., 0, :], ctrl1[..., 1, :]  # [B, K, left, mid, right]
        rotated = torch.stack((cos * a0 + msin * a1, msin * a0 + cos * a1), dim=-2)
        return torch.stack((ctrl0, rotated), dim=3).reshape(b, k, -1)

    # control on axis 5, target on axis 3
    ctrl0 = v[..., 0, :]  # [B, K, left, 2, mid, right] -- untouched
    ctrl1 = v[..., 1, :]
    a0, a1 = ctrl1[:, :, :, 0], ctrl1[:, :, :, 1]  # [B, K, left, mid, right]
    rotated = torch.stack((cos * a0 + msin * a1, msin * a0 + cos * a1), dim=3)
    return torch.stack((ctrl0, rotated), dim=-2).reshape(b, k, -1)


def pqc_param_count(n_io: int, n_mem: int, ansatz: str = "crx_all") -> int:
    """Number of PQC angles ``G`` per neuron for a given ansatz.

    ``crx_all``: ``N*N_mem + N + N_mem`` (3 for (1,1), 5 for (2,1), 8 for
    (2,2)).  ``crx_both`` adds the read-gate layer: ``2*N*N_mem + N + N_mem``.
    """
    if ansatz not in ANSATZE:
        raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {ANSATZE}")
    layers = 2 if ansatz in _TWO_LAYER_ANSATZE else 1
    return layers * n_io * n_mem + n_io + n_mem


def _crx_layer(
    psi: Tensor, theta: Tensor, n_io: int, n_mem: int, io_controls: bool
) -> Tensor:
    """One all-to-all CRX layer, row-major over ``(n, m)``.

    ``io_controls=True`` -> ``CRX(theta[k, n, m])`` with control IO qubit ``n``
    and target memory qubit ``N + m``; ``False`` -> the transpose wiring,
    control memory qubit ``N + m`` and target IO qubit ``n``.
    """
    n_tot = n_io + n_mem
    for n in range(n_io):
        for m in range(n_mem):
            control, target = (n, n_io + m) if io_controls else (n_io + m, n)
            psi = apply_crx(psi, theta[:, n, m], control, target, n_tot)
    return psi


def sqs_step_probabilities(
    psi: Tensor,
    phi: Tensor,
    theta_crx: Tensor,
    theta_rx: Tensor,
    n_io: int,
    n_mem: int,
    ansatz: str = "crx_all",
    theta_crx2: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Apply :math:`U(\mathbf z_t,\boldsymbol\theta)` and return ``(p_joint, psi_pre)``.

    Pure function -- no sampling, no state -- so ``P(outcome | psi_in, x_t, params)``
    can be evaluated for any state, e.g. to check whether it depends on the memory.

    Parameters
    ----------
    psi : ``[B, K, 2**(n_io+n_mem)]`` complex, state at the *start* of the step.
    phi : ``[B, K, n_io]`` real embedding angles (already through Eq. (11)).
    theta_crx : ``[K, n_io, n_mem]`` real CRX angles of the first CRX layer.
    theta_rx : ``[K, n_io + n_mem]`` real RX angles.
    ansatz : ``"crx_all"`` (IO->mem CRX, the published circuit) or ``"crx_both"``
        (IO->mem then the mem->IO read gate).
    theta_crx2 : ``[K, n_io, n_mem]`` angles of the read gate (mem->IO);
        required for ``"crx_both"``, ignored otherwise.

    Returns
    -------
    p_joint : ``[B, K, 2**n_io]`` Born probabilities of the joint IO outcome.
    psi_pre : ``[B, K, 2**(n_io+n_mem)]`` state after the gates, before measurement.
    """
    n_tot = n_io + n_mem
    for n in range(n_io):
        psi = apply_rx(psi, phi[..., n], n, n_tot)

    if ansatz == "crx_all":
        psi = _crx_layer(psi, theta_crx, n_io, n_mem, io_controls=True)
    elif ansatz == "crx_both":
        if theta_crx2 is None:
            raise ValueError("ansatz 'crx_both' needs theta_crx2")
        psi = _crx_layer(psi, theta_crx, n_io, n_mem, io_controls=True)
        psi = _crx_layer(psi, theta_crx2, n_io, n_mem, io_controls=False)
    else:
        raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {ANSATZE}")

    for q in range(n_tot):
        psi = apply_rx(psi, theta_rx[:, q], q, n_tot)
    prob = psi.real * psi.real + psi.imag * psi.imag
    p_joint = prob.reshape(*psi.shape[:2], 1 << n_io, 1 << n_mem).sum(-1)
    return p_joint, psi


# --------------------------------------------------------------------------- #
# layer
# --------------------------------------------------------------------------- #
class SQSLayer(SpikingLayer):
    r"""A bank of ``out_features`` stochastic quantum spiking neurons.

    Each neuron has ``N`` input-output qubits and ``N_mem`` memory qubits and is
    driven by ``N`` weighted input channels from ``in_features`` presynaptic
    neurons.  See the module docstring for the model, the qubit ordering and the
    surrogate-gradient semantics.

    Parameters
    ----------
    in_features : ``P``, number of presynaptic neurons.
    out_features : ``K``, number of neurons in this layer.
    N : number of input-output qubits (= spike channels in and out).
    N_mem : number of memory qubits.
    ansatz : shape of ``V(theta)``, both followed by an ``RX`` on every qubit:
        ``"crx_all"`` (default, the published SQS circuit) all-to-all IO -> memory
        CRX;
        ``"crx_both"`` the same layer followed by the read gate, an all-to-all
        memory -> IO CRX layer with the extra parameter tensor ``theta_crx2``, so
        the memory controls the IO qubits and can change the measurement.
        See :func:`pqc_param_count` for ``G``.
    surrogate : ``"marginal_sigmoid"`` | ``"joint_sigmoid"`` | ``"ste"``.
    w_init : ``"normal"`` -> ``N(0, 1/sqrt(in_features))``;
             ``"kaiming_uniform"`` -> ``U(-1/sqrt(in_features), 1/sqrt(in_features))``.
    w_init_scale : override for the normal std / uniform half-width.
    theta_init : ``"uniform"`` -> ``U(0, pi)``; ``"small"`` -> ``N(0, 0.1)``.
    theta_crx2_scale : initialisation scale of the read gate (``theta_crx2``)
        only.  ``None`` (default) falls back to ``theta_init_scale``.  ``0.0``
        makes the read gate the identity and the circuit exactly ``crx_all``
        (see the module docstring).  Only valid for ``crx_both``.
    theta_crx2_trainable : train ``theta_crx2`` (default) or freeze it at its
        initial value.  Only valid for ``crx_both``.
    dtype : complex dtype of the statevector.  Parameters are stored in the
            matching real dtype (``complex64`` -> ``float32``).

    Shapes
    ------
    ``x_t``    ``[B, N, in_features]`` -> ``spikes`` ``[B, out_features, N]``,
    ``state``  ``[B, out_features, 2**(N+N_mem)]`` complex.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        N: int = 1,
        N_mem: int = 1,
        *,
        ansatz: str = "crx_all",
        surrogate: str = "marginal_sigmoid",
        w_init: str = "normal",
        w_init_scale: float | None = None,
        theta_init: str = "uniform",
        theta_init_scale: float | None = None,
        theta_crx2_scale: float | None = None,
        theta_crx2_trainable: bool = True,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        if N < 1 or N_mem < 1:
            raise ValueError("N and N_mem must be >= 1")
        if ansatz not in ANSATZE:
            raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {ANSATZE}")
        if surrogate not in SURROGATES:
            raise ValueError(f"unknown surrogate {surrogate!r}, expected one of {SURROGATES}")
        if dtype not in _REAL_OF_COMPLEX:
            raise ValueError(f"dtype must be a complex dtype, got {dtype}")

        self.in_features = in_features
        self.out_features = out_features
        self.N = N
        self.N_mem = N_mem
        self.N_tot = N + N_mem
        self.dim = 1 << self.N_tot
        self.dim_io = 1 << N
        self.dim_mem = 1 << N_mem
        self.n_in_channels = N
        self.n_out_channels = N
        self.ansatz = ansatz
        self.surrogate = surrogate
        #: number of PQC angles per neuron
        self.G = pqc_param_count(N, N_mem, ansatz)

        two_layer = ansatz in _TWO_LAYER_ANSATZE
        if not two_layer and (theta_crx2_scale is not None or not theta_crx2_trainable):
            raise ValueError(
                f"ansatz {ansatz!r} has no second CRX layer, so theta_crx2_scale / "
                f"theta_crx2_trainable do not apply; use one of {_TWO_LAYER_ANSATZE}"
            )
        if theta_crx2_scale is not None and theta_crx2_scale < 0.0:
            raise ValueError(f"theta_crx2_scale must be >= 0, got {theta_crx2_scale}")
        #: init scale of the read-gate angles ``theta_crx2``
        self.theta_crx2_scale = None if theta_crx2_scale is None else float(theta_crx2_scale)
        self.theta_crx2_trainable = bool(theta_crx2_trainable)

        rdtype = _REAL_OF_COMPLEX[dtype]
        self.weight = nn.Parameter(torch.empty(N, in_features, out_features, dtype=rdtype))
        self.theta_crx = nn.Parameter(torch.empty(out_features, N, N_mem, dtype=rdtype))
        if two_layer:
            # A frozen read gate is an ``nn.Parameter`` with ``requires_grad=False``, not a
            # buffer, so it stays in ``named_parameters()``, ``state_dict()`` and
            # ``print(layer)``.  ``param_count`` and ``train/surrogate.py::build_optimizer``
            # both skip it, so frozen angles are NOT counted; see ``n_trainable_angles``.
            self.theta_crx2 = nn.Parameter(
                torch.empty(out_features, N, N_mem, dtype=rdtype),
                requires_grad=self.theta_crx2_trainable,
            )
        else:
            self.register_parameter("theta_crx2", None)
        self.theta_rx = nn.Parameter(torch.empty(out_features, self.N_tot, dtype=rdtype))
        self.reset_parameters(w_init, w_init_scale, theta_init, theta_init_scale)
        assert self.param_count() == (
            N * in_features * out_features + self.n_trainable_angles * out_features
        )

        # bit n of the joint IO index j (big-endian): used to turn p_joint into
        # the per-channel marginals P(s_n = 1).
        j = torch.arange(self.dim_io)
        shifts = torch.arange(N - 1, -1, -1)
        self.register_buffer("_bits", ((j[:, None] >> shifts[None, :]) & 1), persistent=False)
        self.register_buffer("_shifts", shifts, persistent=False)

    # ------------------------------------------------------------------ init
    def reset_parameters(
        self,
        w_init: str = "normal",
        w_init_scale: float | None = None,
        theta_init: str = "uniform",
        theta_init_scale: float | None = None,
        theta_crx2_scale: float | None = None,
    ) -> None:
        """(Re)initialise the parameters.

        ``w_init``: ``"normal"`` -> N(0, scale) or ``"kaiming_uniform"`` -> U(-scale, scale),
        scale defaulting to ``1/sqrt(in_features)``.
        ``theta_init``: ``"uniform"`` -> U(0, scale) with scale defaulting to pi;
        ``"small"`` -> N(0, scale) with scale defaulting to 0.1; ``"zeros"``.
        The paper does not state its initialisation.

        ``theta_crx2_scale`` overrides the scale for ``theta_crx2`` alone; ``None``
        falls back to the layer's configured read-gate scale (``self.theta_crx2_scale``) and
        then to ``theta_init_scale``, so a bare ``reset_parameters()`` re-draws the
        layer at the scale it was built with.
        ``theta_crx2`` is drawn LAST so that the weights, ``theta_crx`` and
        ``theta_rx`` come off the same RNG draws at every read-gate scale.
        """
        if theta_crx2_scale is None:
            theta_crx2_scale = self.theta_crx2_scale
        if theta_crx2_scale is not None and theta_crx2_scale < 0.0:
            raise ValueError(f"theta_crx2_scale must be >= 0, got {theta_crx2_scale}")
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            if w_init == "normal":
                self.weight.normal_(0.0, w_init_scale if w_init_scale is not None else bound)
            elif w_init == "kaiming_uniform":
                b = w_init_scale if w_init_scale is not None else bound
                self.weight.uniform_(-b, b)
            else:
                raise ValueError(f"unknown w_init {w_init!r}")

            # (tensor, scale) pairs, theta_crx2 LAST: see the docstring on RNG order
            thetas = [(self.theta_crx, theta_init_scale), (self.theta_rx, theta_init_scale)]
            if self.theta_crx2 is not None:
                thetas.append(
                    (
                        self.theta_crx2,
                        theta_init_scale if theta_crx2_scale is None else theta_crx2_scale,
                    )
                )
            if theta_init == "uniform":
                for theta, scale in thetas:
                    theta.uniform_(0.0, scale if scale is not None else math.pi)
            elif theta_init == "small":
                for theta, scale in thetas:
                    std = scale if scale is not None else 0.1
                    theta.zero_() if std == 0.0 else theta.normal_(0.0, std)
            elif theta_init == "zeros":
                for theta, _ in thetas:
                    theta.zero_()
            else:
                raise ValueError(f"unknown theta_init {theta_init!r}")

    # ------------------------------------------------------------------ misc
    @property
    def n_trainable_angles(self) -> int:
        """PQC angles per neuron that the optimiser can move.

        Equals ``self.G`` unless ``theta_crx2`` is frozen, in which case the
        ``N*N_mem`` read-gate angles drop out.  Parameters are counted as
        ``sum(p.numel() for p in parameters() if p.requires_grad)``, so **frozen
        angles are not counted**: a frozen ``crx_both`` layer reports ``N*N_mem``
        fewer angles per neuron than a trainable one.  The count is the same for
        every frozen read-gate scale, and lower than for ``sqsnn_read_uni``.
        """
        if self.theta_crx2 is None or self.theta_crx2.requires_grad:
            return self.G
        return self.G - self.N * self.N_mem

    @property
    def cdtype(self) -> torch.dtype:
        """Complex dtype of the statevector, tracked from the parameter dtype so
        that ``layer.double()`` promotes the simulation as well."""
        return _COMPLEX_OF_REAL[self.weight.dtype]

    def extra_repr(self) -> str:
        s = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"N={self.N}, N_mem={self.N_mem}, ansatz={self.ansatz!r}, "
            f"surrogate={self.surrogate!r}, G={self.G}"
        )
        if self.theta_crx2 is not None and (
            self.theta_crx2_scale is not None or not self.theta_crx2_trainable
        ):
            s += (
                f", theta_crx2_scale={self.theta_crx2_scale}, "
                f"theta_crx2_trainable={self.theta_crx2_trainable}"
            )
        return s

    def init_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        r""":math:`|0\rangle^{\otimes N_{\rm tot}}` for every sample and neuron."""
        cdtype = self.cdtype
        one = torch.ones(batch_size, self.out_features, 1, dtype=cdtype, device=device)
        rest = torch.zeros(batch_size, self.out_features, self.dim - 1, dtype=cdtype, device=device)
        return torch.cat((one, rest), dim=-1)

    @staticmethod
    def _check_autocast(device_type: str) -> None:
        if torch.is_autocast_enabled(device_type):
            raise RuntimeError(
                "SQSLayer does not support autocast/AMP: complex statevectors must stay "
                "in complex64/complex128. Wrap the call in `torch.autocast(enabled=False)`."
            )

    # ------------------------------------------------------------------ step
    def currents(self, x_t: Tensor) -> Tensor:
        r"""Input currents :math:`z_t^n = \mathbf w^n\mathbf x_t^n` (paper Eq. (10)).

        ``x_t`` ``[B, N, in_features]`` -> ``z`` ``[B, out_features, N]``.
        """
        if x_t.dim() != 3 or x_t.shape[1] != self.N or x_t.shape[2] != self.in_features:
            raise ValueError(
                f"x_t must be [B, {self.N}, {self.in_features}], got {tuple(x_t.shape)}"
            )
        x = x_t.to(self.weight.dtype)
        if self.N == 1:  # fast path: a single matmul
            return torch.matmul(x[:, 0], self.weight[0]).unsqueeze(-1)
        return torch.einsum("bnp,npk->bkn", x, self.weight)

    def angles(self, z: Tensor) -> Tensor:
        r"""Embedding angles (paper Eq. (11)): ``phi = pi*z`` if ``z > 0`` else 0."""
        return torch.where(z > 0, math.pi * z, torch.zeros((), dtype=z.dtype, device=z.device))

    def probabilities(self, x_t: Tensor, psi: Tensor) -> tuple[Tensor, Tensor]:
        """Gates only: return ``(p_joint [B, K, 2**N], psi_pre [B, K, 2**N_tot])``.

        No sampling and no collapse, so this can be called repeatedly on any
        state without touching the RNG.
        """
        self._check_autocast(psi.device.type)
        phi = self.angles(self.currents(x_t))
        return sqs_step_probabilities(
            psi.to(self.cdtype),
            phi,
            self.theta_crx,
            self.theta_rx,
            self.N,
            self.N_mem,
            self.ansatz,
            self.theta_crx2,
        )

    def _collapse(self, psi_pre: Tensor, outcome: Tensor, p_sel: Tensor) -> Tensor:
        r"""Project the IO qubits onto ``outcome`` and reset them to ``|0..0>``.

        ``psi_next = |0..0>_IO (x) (<s| (x) I)|psi_pre> / sqrt(p_sel)`` -- paper
        Eqs. (15)-(16).  Built with ``cat`` so nothing is written in place.
        """
        b, k, _ = psi_pre.shape
        blocks = psi_pre.reshape(b, k, self.dim_io, self.dim_mem)
        idx = outcome.reshape(b, k, 1, 1).expand(b, k, 1, self.dim_mem)
        mem = blocks.gather(2, idx).squeeze(2)  # [B, K, 2**N_mem]
        mem = mem / torch.sqrt(p_sel.clamp_min(1e-12)).unsqueeze(-1)
        pad = torch.zeros(b, k, self.dim - self.dim_mem, dtype=mem.dtype, device=mem.device)
        return torch.cat((mem, pad), dim=-1)

    def forward_with_outcome(
        self, x_t: Tensor, psi: Tensor, outcome: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Teacher forcing: apply the step and collapse onto a *given* outcome.

        Parameters
        ----------
        outcome : ``[B, K]`` int64 joint IO outcome index (big-endian bits).

        Returns
        -------
        ``(p_sel [B, K], psi_next [B, K, 2**N_tot])`` -- no sampling anywhere, so
        ``log p_sel`` is a clean differentiable likelihood.
        """
        p_joint, psi_pre = self.probabilities(x_t, psi)
        outcome = outcome.to(torch.int64)
        if outcome.shape != p_joint.shape[:2]:
            outcome = outcome.expand(p_joint.shape[:2])
        p_sel = p_joint.gather(-1, outcome.unsqueeze(-1)).squeeze(-1)
        return p_sel, self._collapse(psi_pre, outcome, p_sel)

    def surrogate_spikes(self, s_hard: Tensor, p: Tensor, p_sel: Tensor) -> Tensor:
        """Straight-through spikes: hard value forward, Appendix-A slope backward."""
        if self.surrogate == "marginal_sigmoid":
            soft = torch.sigmoid(p)
        elif self.surrogate == "joint_sigmoid":
            soft = torch.sigmoid(p_sel).unsqueeze(-1).expand_as(s_hard)
        elif self.surrogate == "ste":
            soft = p
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(self.surrogate)
        return s_hard + (soft - soft.detach())

    def forward(
        self, x_t: Tensor, psi: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """One SQS time step with a single-shot measurement.

        Parameters
        ----------
        x_t : ``[B, N, in_features]`` presynaptic signals at time ``t``.
        psi : ``[B, out_features, 2**N_tot]`` complex state (from
              :meth:`init_state` or the previous step).

        Returns
        -------
        spikes : ``[B, out_features, N]`` float, hard 0/1 with the surrogate
                 gradient attached.
        psi_next : ``[B, out_features, 2**N_tot]`` post-measurement state.
        aux : ``{"p", "p_joint", "outcome", "p_sel", "psi_in", "psi_pre", "z"}``.
        """
        self._check_autocast(psi.device.type)
        psi_in = psi.to(self.cdtype)
        z = self.currents(x_t)
        phi = self.angles(z)
        p_joint, psi_pre = sqs_step_probabilities(
            psi_in,
            phi,
            self.theta_crx,
            self.theta_rx,
            self.N,
            self.N_mem,
            self.ansatz,
            self.theta_crx2,
        )

        # per-channel Born marginals P(s_n = 1)
        p = torch.matmul(p_joint, self._bits.to(p_joint.dtype))

        # one single-shot joint sample per (sample, neuron) via inverse CDF
        with torch.no_grad():
            probs = p_joint.detach()
            cdf = probs.cumsum(-1)
            u = torch.rand(probs.shape[0], probs.shape[1], 1, dtype=probs.dtype, device=probs.device)
            outcome = torch.searchsorted(cdf.contiguous(), u.contiguous()).squeeze(-1)
            outcome = outcome.clamp_(max=self.dim_io - 1)

        p_sel = p_joint.gather(-1, outcome.unsqueeze(-1)).squeeze(-1)
        s_hard = ((outcome.unsqueeze(-1) >> self._shifts) & 1).to(p.dtype)
        spikes = self.surrogate_spikes(s_hard, p, p_sel)
        psi_next = self._collapse(psi_pre, outcome, p_sel)

        aux = {
            "p": p,
            "p_joint": p_joint,
            "outcome": outcome,
            "p_sel": p_sel,
            "psi_in": psi_in,
            "psi_pre": psi_pre,
            "z": z,
        }
        return spikes, psi_next, aux
