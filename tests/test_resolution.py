"""Config resolution: where T, the encoder, hidden, lr and epochs come from, and model building.

The dataset must be able to overrule the protocol on ``T`` and ``encoder`` (the memory probe
brings its own time axis), ``dataset.protocol_overrides`` beats both, and every value lands
in ``cfg.resolved``.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from sqs_bench.config import resolve_cfg, resolved_hidden
from sqs_bench.models.build import build_model, hidden_widths, locate
from sqs_bench.neurons.qlif import QLIFLayer
from sqs_bench.neurons.sqs import SQSLayer


# ------------------------------------------------------------------------- T
def test_memprobe_T_comes_from_the_sum_resolver(make_cfg):
    cfg = resolve_cfg(make_cfg())
    assert cfg.resolved.T == 2 + 8 + 3
    assert cfg.protocol.T == cfg.resolved.T  # written back, so config.yaml reads true


@pytest.mark.parametrize("lead_in", [0, 1])
@pytest.mark.parametrize("delay", [1, 2, 4, 8, 16])
def test_memprobe_T_tracks_the_delay_and_the_lead_in(make_cfg, delay, lead_in):
    cfg = resolve_cfg(make_cfg([f"dataset.delay={delay}", f"dataset.lead_in={lead_in}"]))
    assert cfg.resolved.T == lead_in + 1 + delay + 1 + 3


def test_protocol_T_is_the_fallback(make_cfg):
    cfg = resolve_cfg(make_cfg(["dataset.T=null"]))
    assert cfg.resolved.T == 5  # conf/protocol/sqs_ce.yaml


def test_protocol_overrides_beat_dataset_T(make_cfg):
    """``protocol_overrides.<p>.T`` is the most specific place T can be set."""
    cfg = make_cfg(["++dataset.protocol_overrides.sqs_ce.T=14"])
    assert resolve_cfg(cfg).resolved.T == 14


# ------------------------------------------------------------------- encoder
def test_memprobe_uses_the_identity_encoder_without_a_deviation(make_cfg):
    cfg = resolve_cfg(make_cfg())
    assert cfg.resolved.encoder == "identity"
    assert not any("overrides the protocol" in d for d in cfg.deviations)


def test_dataset_encoder_beats_the_protocol(make_cfg):
    cfg = resolve_cfg(make_cfg(["protocol.encoder=other"]))
    assert cfg.resolved.encoder == "identity"
    assert any("overrides the protocol's 'other'" in d for d in cfg.deviations)


def test_protocol_overrides_beat_dataset_encoder(make_cfg):
    cfg = make_cfg(["++dataset.protocol_overrides.sqs_ce.encoder=other"])
    assert resolve_cfg(cfg).resolved.encoder == "other"


def test_an_encoder_is_required(make_cfg):
    with pytest.raises(ValueError, match="no encoder"):
        resolve_cfg(make_cfg(["dataset.encoder=null", "protocol.encoder=null"]))


# -------------------------------------------------------------------- hidden
def test_memprobe_hidden_is_64(make_cfg):
    cfg = resolve_cfg(make_cfg())
    assert resolved_hidden(cfg) == 64 and isinstance(resolved_hidden(cfg), int)
    assert cfg.model.hidden == 64


def test_model_hidden_beats_the_dataset(make_cfg):
    cfg = resolve_cfg(make_cfg(["model.hidden=8"]))
    assert resolved_hidden(cfg) == 8


def test_protocol_hidden_is_the_fallback(make_cfg):
    cfg = resolve_cfg(make_cfg(["dataset.protocol_overrides.sqs_ce.hidden=null",
                                "protocol.hidden=32"]))
    assert resolved_hidden(cfg) == 32


def test_hidden_null_everywhere_is_an_error(make_cfg):
    with pytest.raises(ValueError, match="hidden is null"):
        resolve_cfg(make_cfg(["dataset.protocol_overrides.sqs_ce.hidden=null"]))


# ----------------------------------------------------------- lr, epochs, batch
def test_lr_precedence(make_cfg):
    assert resolve_cfg(make_cfg()).resolved.lr == pytest.approx(5e-4)
    assert resolve_cfg(
        make_cfg(["++model.lr_by_protocol.sqs_ce=0.01"])
    ).resolved.lr == pytest.approx(0.01)
    assert resolve_cfg(
        make_cfg(["++model.lr_by_protocol.sqs_ce=0.01", "train.lr=0.2"])
    ).resolved.lr == pytest.approx(0.2)


def test_epochs_precedence(make_cfg):
    assert resolve_cfg(make_cfg()).resolved.epochs == 15
    assert resolve_cfg(make_cfg(["++train.epochs_by_dataset.memprobe=60"])).resolved.epochs == 60
    assert resolve_cfg(
        make_cfg(["++train.epochs_by_dataset.memprobe=60", "train.epochs=150"])
    ).resolved.epochs == 150


def test_batch_size_precedence(make_cfg):
    assert resolve_cfg(make_cfg()).resolved.batch_size == 128
    assert resolve_cfg(make_cfg(["train.batch_size=16"])).resolved.batch_size == 16


def test_every_model_resolves(make_cfg):
    for model in ("lif", "qlif", "sqsnn", "sqsnn_uni", "sqsnn_read", "sqsnn_read_uni",
                  "sqsnn_angle0", "sqsnn_angle_pi4", "sqsnn_angle_pi2", "sqsnn_angle_pi"):
        cfg = resolve_cfg(make_cfg([f"model={model}", "variant=x_delay8"]))
        assert cfg.resolved.model == model
        assert cfg.resolved.hidden == 64 and cfg.resolved.lr == pytest.approx(5e-4)
        assert list(cfg.deviations) == []


# ------------------------------------------------------------------ hidden_widths
@pytest.mark.parametrize(("hidden", "expected"), [(None, []), (0, []), (8, [8]), (64, [64])])
def test_hidden_widths(hidden, expected):
    assert hidden_widths(hidden) == expected


# ---------------------------------------------------------------- build_model
def sqs_model_cfg(**layer):
    return OmegaConf.create(
        {
            "name": "sqsnn",
            "layer": {"_target_": "sqs_bench.neurons.sqs.SQSLayer", "N": 1, "N_mem": 1, **layer},
            "hidden_layer_overrides": {},
            "output_layer_overrides": {},
        }
    )


def test_locate():
    assert locate("sqs_bench.neurons.sqs.SQSLayer") is SQSLayer
    assert locate("sqs_bench.neurons.qlif.QLIFLayer") is QLIFLayer
    with pytest.raises(ValueError):
        locate("SQSLayer")


@pytest.mark.parametrize(
    ("hidden", "sizes"), [(0, [(5, 4)]), (None, [(5, 4)]), (8, [(5, 8), (8, 4)])]
)
def test_build_model_layer_shapes(hidden, sizes):
    model = build_model(sqs_model_cfg(), in_features=5, n_classes=4, hidden=hidden)
    assert [(l.in_features, l.out_features) for l in model.layers] == sizes


def test_build_model_uses_the_right_overrides_per_layer(make_cfg):
    """The hidden layer gets ``hidden_layer_overrides``, the output layer the output ones."""
    cfg = make_cfg(["model=qlif"]).model
    cfg.hidden_layer_overrides = {"threshold": 0.25}
    cfg.output_layer_overrides = {"threshold": 0.5}
    model = build_model(cfg, in_features=5, n_classes=4, hidden=8)
    assert [l.threshold for l in model.layers] == [0.25, 0.5]
    assert [l.tau_scale for l in model.layers] == [200.0, 200.0]


def test_build_model_drops_unsupported_kwargs():
    """``bias`` means nothing to an SQS layer: it is dropped and the drop is reported."""
    dropped: list[str] = []
    model = build_model(sqs_model_cfg(bias=True), in_features=5, n_classes=4, hidden=8,
                        dropped=dropped)
    assert len(model.layers) == 2
    assert dropped and all("bias" in d for d in dropped)


def test_build_model_forwards_the_layer_kwargs():
    model = build_model(sqs_model_cfg(ansatz="crx_both", theta_init="small"),
                        in_features=5, n_classes=4, hidden=8)
    assert all(l.ansatz == "crx_both" for l in model.layers)
    assert all(l.theta_crx2 is not None for l in model.layers)
    out = model((torch.rand(3, 2, 5) < 0.5).float())
    assert out.spikes_out.shape == (3, 2, 4, 1)


def test_build_model_needs_a_layer_block():
    with pytest.raises(ValueError, match="layer"):
        build_model(OmegaConf.create({"name": "x"}), in_features=5, n_classes=4, hidden=8)
