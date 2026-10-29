"""The frozen read-gate angle: one circuit, four read strengths.

``crx_both`` scores above chance on the delayed-recall task and ``crx_all`` does not, but
``crx_both`` adds more than a memory -> IO coupling: it also adds one trainable angle and one
CRX layer per neuron.
The angle runs separate the coupling from the rest.  They hold the architecture at ``crx_both``
-- same wiring, same gate count, same parameter tensors -- and vary only the *strength* of the
read gate through ``theta_crx2_scale``, the initialisation scale of the second CRX layer
alone, with those angles frozen so the optimiser cannot undo the setting.

What these tests pin, in the order the argument needs them:

a. ``theta_crx2_scale=0.0`` is exactly memory-blind and numerically **equals** ``crx_all`` on
   the same inputs, so the scale-0 arm is a check on the task as well as an arm.
b. Readout sensitivity rises **monotonically** across the four scales (median over 24 random
   layers, and pairwise per layer).
c. ``n_params`` and the trainable-parameter count are identical at every scale, so the arms
   differ in the read coupling and in nothing the optimiser can spend.
d. ``theta_crx2`` gets no gradient and does not move under an optimiser at any scale.

The scales are 0, pi/4, pi/2 and pi, matching ``conf/model/sqsnn_angle{0,_pi4,_pi2,_pi}.yaml``.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import sqs_bench
from sqs_bench.neurons.sqs import SQSLayer, sqs_step_probabilities

MODEL_CONF = Path(sqs_bench.__file__).parent / "conf" / "model"

#: the four arms: (config name, theta_crx2_scale)
ARMS: list[tuple[str, float]] = [
    ("sqsnn_angle0", 0.0),
    ("sqsnn_angle_pi4", 0.25 * math.pi),
    ("sqsnn_angle_pi2", 0.5 * math.pi),
    ("sqsnn_angle_pi", 1.0 * math.pi),
]
SCALES: list[float] = [s for _, s in ARMS]


def _angle_layer(scale, *, in_features=3, out_features=1, n_io=1, n_mem=1, trainable=False):
    """A ``crx_both`` layer at one read-gate scale, everything else as ``sqsnn_read_uni``."""
    return SQSLayer(
        in_features,
        out_features,
        n_io,
        n_mem,
        ansatz="crx_both",
        theta_init="uniform",
        theta_crx2_scale=scale,
        theta_crx2_trainable=trainable,
        dtype=torch.complex128,
    )


def _readout_sensitivity(layer, x, states=8, seed=0):
    """Spread of the Born probabilities over random memory states -- the readout signal.

    Same measure as ``tests/test_sqs_trajectory.py``: the IO register starts in ``|0...0>`` and
    only the memory block varies, so a non-zero spread means the memory reaches the measurement.
    """
    dim, dim_mem = layer.dim, layer.dim_mem
    g = torch.Generator().manual_seed(seed)
    probs = []
    for _ in range(states):
        m = torch.randn(dim_mem, generator=g, dtype=torch.float64).to(torch.complex128)
        m = m + 1j * torch.randn(dim_mem, generator=g, dtype=torch.float64)
        m = m / m.norm()
        psi = torch.zeros(1, layer.out_features, dim, dtype=torch.complex128)
        psi[0, :, :dim_mem] = m
        p_joint, _ = layer.probabilities(x, psi)
        probs.append(p_joint.detach().reshape(-1))
    stacked = torch.stack(probs)
    return (stacked.max(0).values - stacked.min(0).values).max().item()


def _paired_sensitivities(n_layers=24, seed0=1000):
    """Sensitivity of every scale on ``n_layers`` *paired* layers, ``[n_layers, 4]``.

    Paired because ``theta_crx2`` is drawn LAST in ``reset_parameters``: at a fixed torch seed the
    weights, ``theta_crx`` and ``theta_rx`` are bit-identical across scales and ``theta_crx2`` is
    literally ``scale * u`` with the same ``u``.  So a row of this table is one circuit seen at
    four coupling strengths and nothing else varies.
    """
    rows = []
    for i in range(n_layers):
        row = []
        for scale in SCALES:
            torch.manual_seed(seed0 + i)
            layer = _angle_layer(scale)
            with torch.no_grad():
                layer.weight.abs_().add_(0.3)  # drive the RX embedding past Eq. (11)'s kink
            torch.manual_seed(seed0 + i)  # the same drive at every scale too
            x = torch.rand(1, layer.N, layer.in_features, dtype=torch.float64)
            row.append(_readout_sensitivity(layer, x, seed=seed0 + i))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# a. scale 0 is crx_all
# --------------------------------------------------------------------------- #
def test_scale0_angles_are_exactly_zero():
    """``U(0, 0)`` is a point mass at 0, so the second CRX layer is literally ``CRX(0) = I``."""
    for n_io, n_mem in ((1, 1), (2, 1), (2, 2)):
        torch.manual_seed(0)
        layer = _angle_layer(0.0, out_features=5, n_io=n_io, n_mem=n_mem)
        assert layer.theta_crx2.abs().max().item() == 0.0, (n_io, n_mem)
        assert layer.theta_crx.abs().max().item() > 0.0, "the write layer must still be random"


@pytest.mark.parametrize("n_io,n_mem", [(1, 1), (2, 1), (2, 2)])
def test_scale0_is_numerically_identical_to_crx_all(n_io, n_mem):
    """Scale 0 is not *like* the paper's circuit, it **is** the paper's circuit.

    Same weights, same ``theta_crx``, same ``theta_rx``, same input, same state -> bit-for-bit the
    same Born probabilities and the same pre-measurement statevector as ``ansatz="crx_all"``.
    Checked on a random (entangled) input state, which is stronger than the ``|0..0>`` reset the
    layer actually uses.
    """
    torch.manual_seed(17 + n_io + 5 * n_mem)
    b, k, p = 3, 4, 5
    angle0 = _angle_layer(0.0, in_features=p, out_features=k, n_io=n_io, n_mem=n_mem)
    plain = SQSLayer(p, k, n_io, n_mem, ansatz="crx_all", dtype=torch.complex128)
    with torch.no_grad():
        for name in ("weight", "theta_crx", "theta_rx"):
            getattr(plain, name).copy_(getattr(angle0, name))

    x = torch.rand(b, n_io, p, dtype=torch.float64)
    psi = torch.randn(b, k, 2 ** (n_io + n_mem), dtype=torch.complex128)
    psi = psi / psi.norm(dim=-1, keepdim=True)

    p_angle, psi_angle = angle0.probabilities(x, psi)
    p_all, psi_all = plain.probabilities(x, psi)
    assert torch.equal(p_angle, p_all), (p_angle - p_all).abs().max().item()
    assert torch.equal(psi_angle, psi_all), (psi_angle - psi_all).abs().max().item()


@pytest.mark.parametrize("theta_init", ["uniform", "small"])
def test_scale0_is_exactly_memory_blind(theta_init):
    """The falsifiable prediction of the angle runs.

    At scale 0 the circuit is ``crx_all``, so at ``N = N_mem = 1`` no output probability depends
    on the memory state -- to floating point.  A scale-0 run that scores above chance on the
    recall task would mean the task leaks the label.
    """
    worst = 0.0
    for i in range(20):
        torch.manual_seed(300 + i)
        layer = SQSLayer(
            3,
            1,
            1,
            1,
            ansatz="crx_both",
            theta_init=theta_init,
            theta_crx2_scale=0.0,
            theta_crx2_trainable=False,
            dtype=torch.complex128,
        )
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)
        x = torch.rand(1, 1, 3, dtype=torch.float64)
        worst = max(worst, _readout_sensitivity(layer, x, seed=300 + i))
    assert worst < 1e-12, (theta_init, worst)


def test_scale0_has_no_credit_path_across_time():
    """Blindness is not only a static property: at scale 0 no gradient reaches a past input.

    The Born-probability spread is measured at one step; this walks five steps and asks whether
    BPTT can carry credit from an earlier input to a later output.  ``crx_all`` fails this and so
    must scale 0; scale pi must clear it by many orders of magnitude.
    """

    def reach(scale, source_step, T=5, seed=41):
        torch.manual_seed(seed)
        layer = _angle_layer(scale, in_features=4, out_features=2)
        with torch.no_grad():
            layer.weight.abs_().add_(0.3)
        xs = [
            torch.rand(8, 1, 4, dtype=torch.float64, requires_grad=(t == source_step))
            for t in range(T)
        ]
        psi = layer.init_state(8, "cpu").to(torch.complex128)
        for t in range(T):
            _, psi, aux = layer.forward(xs[t], psi)
        aux["p"].sum().backward()
        return xs[source_step].grad.abs().max().item()

    for source in range(4):
        assert reach(0.0, source) < 1e-12, source
        assert reach(math.pi, source) > 1e-3, source


def test_scale0_matches_a_hand_built_crx_all_call():
    """The same anchor at the level of the pure gate function: feeding ``theta_crx2 = 0`` to
    ``crx_both`` is the same call as ``crx_all`` with no second tensor."""
    torch.manual_seed(0)
    for n_io, n_mem in ((1, 1), (2, 1), (2, 2)):
        dim = 2 ** (n_io + n_mem)
        psi = torch.randn(3, 2, dim, dtype=torch.complex128)
        psi = psi / psi.norm(dim=-1, keepdim=True)
        phi = torch.rand(3, 2, n_io, dtype=torch.float64)
        a = torch.rand(2, n_io, n_mem, dtype=torch.float64) * 3
        zero = torch.zeros(2, n_io, n_mem, dtype=torch.float64)
        rx = torch.rand(2, n_io + n_mem, dtype=torch.float64)
        p_angle, _ = sqs_step_probabilities(psi, phi, a, rx, n_io, n_mem, "crx_both", zero)
        p_all, _ = sqs_step_probabilities(psi, phi, a, rx, n_io, n_mem, "crx_all", None)
        assert torch.equal(p_angle, p_all), (n_io, n_mem)


# --------------------------------------------------------------------------- #
# b. readout sensitivity rises with the scale
# --------------------------------------------------------------------------- #
def test_readout_sensitivity_increases_monotonically_with_the_scale():
    """The scale does what it says: more coupling, more memory in the readout.

    Median over 24 random layers, and -- because the layers are paired at a fixed seed -- also
    layer by layer, so both the aggregate and the paired statement are asserted.
    """
    table = _paired_sensitivities()
    med = table.median(dim=0).values

    # paired: every single one of the 24 layers is monotone in the scale
    diffs = table[:, 1:] - table[:, :-1]
    assert (diffs > 0).all(), table.tolist()

    assert med[0].item() < 1e-12, med.tolist()
    for lo, hi in zip(range(3), range(1, 4)):
        assert med[hi].item() > med[lo].item(), (lo, hi, med.tolist())
    # a real step, not floating-point noise: the medians are about
    # 4.4e-16, 0.113, 0.217, 0.406 -- about x1.9 per arm
    assert med[1].item() > 1e-2, med.tolist()
    assert med[2].item() > 1.5 * med[1].item(), med.tolist()
    assert med[3].item() > 1.5 * med[2].item(), med.tolist()


def test_the_scale_is_the_only_thing_that_changes_across_arms():
    """The pairing the monotonicity test relies on.

    ``theta_crx2`` is drawn last, so at one seed the weights and the other two angle tensors are
    bit-identical across scales and ``theta_crx2`` is exactly ``scale * u`` with a shared ``u``.
    Without this the four arms would differ in their whole initialisation, not just the scale.
    """
    layers = []
    for scale in SCALES:
        torch.manual_seed(7)
        layers.append(_angle_layer(scale, in_features=6, out_features=5))
    ref = layers[-1]
    for layer in layers[:-1]:
        for name in ("weight", "theta_crx", "theta_rx"):
            assert torch.equal(getattr(layer, name), getattr(ref, name)), name
    u = ref.theta_crx2 / math.pi
    for layer, scale in zip(layers, SCALES):
        assert torch.allclose(layer.theta_crx2, scale * u, atol=1e-12), scale


# --------------------------------------------------------------------------- #
# c. capacity is constant across the arms
# --------------------------------------------------------------------------- #
def test_param_count_is_identical_across_scales():
    """``n_params`` -- the number every run reports -- must not move."""
    counts, trainable, total = set(), set(), set()
    for scale in SCALES:
        torch.manual_seed(0)
        layer = _angle_layer(scale, in_features=9, out_features=7)
        counts.add(layer.param_count())
        trainable.add(sum(p.numel() for p in layer.parameters() if p.requires_grad))
        total.add(sum(p.numel() for p in layer.parameters()))
    assert len(counts) == 1, counts
    assert len(trainable) == 1, trainable
    assert counts == trainable, (counts, trainable)
    assert len(total) == 1, total  # the tensors themselves are the same size too
    assert counts.pop() == 1 * 9 * 7 + 3 * 7  # weights + (G - frozen) angles per neuron


def test_shapes_and_gate_count_are_identical_across_scales():
    """Same circuit, four strengths: same ``G``, same tensors, same gate count."""
    ref = _angle_layer(SCALES[0], in_features=4, out_features=3)
    for scale in SCALES[1:]:
        layer = _angle_layer(scale, in_features=4, out_features=3)
        assert layer.G == ref.G == 4
        assert layer.ansatz == ref.ansatz == "crx_both"
        for name in ("weight", "theta_crx", "theta_crx2", "theta_rx"):
            assert getattr(layer, name).shape == getattr(ref, name).shape, name
        assert {n for n, _ in layer.named_parameters()} == {n for n, _ in ref.named_parameters()}


def test_frozen_angles_are_not_counted():
    """The counter's actual behaviour, pinned so nobody has to re-derive it.

    ``param_count`` filters on ``requires_grad``, so freezing ``theta_crx2`` REMOVES ``N*N_mem``
    angles per neuron from the reported count.  Identical across scales -- the comparison the
    angle runs need -- but an angle arm is NOT parameter-matched to a trainable ``crx_both``
    model such as ``sqsnn_read_uni``.
    """
    k = 6
    frozen = _angle_layer(math.pi, in_features=4, out_features=k, trainable=False)
    trained = _angle_layer(math.pi, in_features=4, out_features=k, trainable=True)
    assert frozen.G == trained.G == 4
    assert frozen.n_trainable_angles == 3
    assert trained.n_trainable_angles == 4
    assert trained.param_count() - frozen.param_count() == k
    # ... and the tensor is still there, just not counted: a buffer would hide it entirely
    assert "theta_crx2" in dict(frozen.named_parameters())
    assert "theta_crx2" in frozen.state_dict()
    assert "theta_crx2_scale=3.14" in repr(frozen)


# --------------------------------------------------------------------------- #
# d. the setting cannot be trained away
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", SCALES)
def test_theta_crx2_gets_no_gradient(scale):
    """A backward pass through several steps leaves ``theta_crx2`` without a gradient, while
    every trainable tensor gets a finite non-zero one (so the layer is genuinely training)."""
    torch.manual_seed(23)
    T, b, k, p = 3, 4, 2, 3
    layer = _angle_layer(scale, in_features=p, out_features=k)
    with torch.no_grad():
        layer.weight.abs_().add_(0.3)
    xs = torch.rand(T, b, 1, p, dtype=torch.float64)
    psi = layer.init_state(b, "cpu").to(torch.complex128)
    loss = torch.zeros((), dtype=torch.float64)
    for t in range(T):
        spikes, psi, aux = layer.forward(xs[t], psi)
        loss = loss + aux["p"].sum() + spikes.sum()
    loss.backward()

    assert layer.theta_crx2.requires_grad is False
    assert layer.theta_crx2.grad is None
    for name in ("weight", "theta_crx", "theta_rx"):
        g = getattr(layer, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0.0, name


@pytest.mark.parametrize("scale", SCALES)
def test_theta_crx2_does_not_move_under_an_optimiser(scale):
    """Ten Adam steps over the parameters ``train/surrogate.py::build_optimizer`` would hand it
    (it keeps only ``requires_grad``): the read gate is bit-for-bit unchanged and the trainable
    angles have moved."""
    torch.manual_seed(29)
    T, b, k, p = 3, 5, 2, 4
    layer = _angle_layer(scale, in_features=p, out_features=k)
    with torch.no_grad():
        layer.weight.abs_().add_(0.3)
    before = layer.theta_crx2.detach().clone()
    theta_crx_before = layer.theta_crx.detach().clone()

    params = [q for q in layer.parameters() if q.requires_grad]
    assert len(params) == 3, [n for n, q in layer.named_parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=0.1)
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        xs = torch.rand(T, b, 1, p, dtype=torch.float64)
        psi = layer.init_state(b, "cpu").to(torch.complex128)
        loss = torch.zeros((), dtype=torch.float64)
        for t in range(T):
            _, psi, aux = layer.forward(xs[t], psi)
            loss = loss + aux["p"].sum()
        loss.backward()
        opt.step()

    assert torch.equal(layer.theta_crx2, before), (
        scale,
        (layer.theta_crx2 - before).abs().max().item(),
    )
    assert not torch.equal(layer.theta_crx, theta_crx_before), "the layer did not train at all"


def test_build_optimizer_skips_the_frozen_read_gate(make_cfg):
    """The real trainer's optimiser, built from the real config, never sees ``theta_crx2``."""
    from sqs_bench.config import resolve_cfg
    from sqs_bench.models.build import build_model
    from sqs_bench.train.surrogate import build_optimizer

    cfg = resolve_cfg(make_cfg(["model=sqsnn_angle_pi", "variant=angle_delay8"]))
    model = build_model(cfg.model, in_features=5, n_classes=4, hidden=8)
    opt = build_optimizer(cfg, model)
    handed = {id(p) for group in opt.param_groups for p in group["params"]}
    for name, p in model.named_parameters():
        assert (id(p) in handed) == (not name.endswith("theta_crx2")), name


def test_reset_parameters_keeps_the_scale():
    """A bare ``reset_parameters()`` must re-draw the layer AT ITS SCALE.

    The obvious bug would be for a re-init to fall back to ``theta_init_scale`` and silently push
    every arm to full strength; then the four arms would all be the same.
    """
    for scale in SCALES:
        torch.manual_seed(5)
        layer = _angle_layer(scale, in_features=4, out_features=6)
        layer.reset_parameters(theta_init="uniform")
        assert layer.theta_crx2.max().item() <= scale + 1e-12, scale
        if scale == 0.0:
            assert layer.theta_crx2.abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
# guards on the knobs themselves
# --------------------------------------------------------------------------- #
def test_defaults_are_untouched():
    """``theta_crx2_scale=None`` and ``theta_crx2_trainable=True`` give the same layer as
    leaving them out, so every model without the two keys is unaffected."""
    for ansatz in ("crx_all", "crx_both"):
        torch.manual_seed(101)
        a = SQSLayer(5, 4, 1, 1, ansatz=ansatz, theta_init="uniform")
        torch.manual_seed(101)
        if ansatz == "crx_both":
            b = SQSLayer(
                5, 4, 1, 1, ansatz=ansatz, theta_init="uniform",
                theta_crx2_scale=None, theta_crx2_trainable=True,
            )
        else:
            b = SQSLayer(5, 4, 1, 1, ansatz=ansatz, theta_init="uniform")
        for name, param in a.named_parameters():
            assert torch.equal(param, dict(b.named_parameters())[name]), (ansatz, name)
            assert param.requires_grad, (ansatz, name)
        assert a.param_count() == b.param_count()
        assert a.n_trainable_angles == a.G


def test_angle_knobs_are_rejected_for_crx_all():
    """Silently ignoring the setting on an ansatz with no second CRX layer would produce four
    identical arms that look like a flat response.  Refuse instead."""
    with pytest.raises(ValueError, match="no second CRX layer"):
        SQSLayer(3, 2, 1, 1, ansatz="crx_all", theta_crx2_scale=1.0)
    with pytest.raises(ValueError, match="no second CRX layer"):
        SQSLayer(3, 2, 1, 1, ansatz="crx_all", theta_crx2_trainable=False)


def test_negative_scale_is_rejected():
    with pytest.raises(ValueError, match="theta_crx2_scale must be >= 0"):
        SQSLayer(3, 2, 1, 1, ansatz="crx_both", theta_crx2_scale=-0.1)


# --------------------------------------------------------------------------- #
# the four config files are the four arms
# --------------------------------------------------------------------------- #
def test_configs_declare_the_four_scales():
    """The YAML is the experiment; keep it in step with this file.

    Every arm must be ``crx_both`` at ``N = N_mem = 1`` with a frozen ``theta_crx2``, must differ
    from ``sqsnn_read_uni`` in the two ``theta_crx2`` keys and nothing else, and the four scales
    must be 0, pi/4, pi/2, pi in that order.
    """

    def load(name):
        return OmegaConf.to_container(OmegaConf.load(MODEL_CONF / f"{name}.yaml"))

    base = load("sqsnn_read_uni")
    for (name, scale), expect in zip(ARMS, SCALES):
        cfg = load(name)
        assert cfg["name"] == name
        layer = cfg["layer"]
        assert layer["ansatz"] == "crx_both"
        assert layer["N"] == layer["N_mem"] == 1
        assert layer["theta_init"] == "uniform"
        assert layer["theta_crx2_trainable"] is False
        assert layer["theta_crx2_scale"] == pytest.approx(expect, abs=1e-12)
        # identical to sqsnn_read_uni apart from the name and the two theta_crx2 keys
        assert {k: v for k, v in cfg.items() if k != "name" and k != "layer"} == {
            k: v for k, v in base.items() if k != "name" and k != "layer"
        }
        assert {k: v for k, v in layer.items() if not k.startswith("theta_crx2")} == {
            k: v for k, v in base["layer"].items() if not k.startswith("theta_crx2")
        }
        # and the config is instantiable as written
        built = SQSLayer(
            4,
            3,
            layer["N"],
            layer["N_mem"],
            ansatz=layer["ansatz"],
            surrogate=layer["surrogate"],
            w_init=layer["w_init"],
            theta_init=layer["theta_init"],
            theta_init_scale=layer["theta_init_scale"],
            theta_crx2_scale=layer["theta_crx2_scale"],
            theta_crx2_trainable=layer["theta_crx2_trainable"],
        )
        assert built.theta_crx2.requires_grad is False
        assert built.theta_crx2.max().item() <= scale + 1e-6
