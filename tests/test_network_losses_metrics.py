"""Network plumbing, the loss and the metrics, on real SQS and QLIF layers."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sqs_bench.losses import membrane_ce_loss
from sqs_bench.metrics import (
    RunMetrics,
    accuracy,
    confusion_matrix,
    layer_rates,
    membrane_decode,
    per_neuron_rate,
    rate_decode,
    spikes_per_step,
)
from sqs_bench.models.feedforward import FeedForwardSNN
from sqs_bench.neurons.qlif import QLIFLayer
from sqs_bench.neurons.sqs import SQSLayer
from sqs_bench.train.loops import compute_loss, logits_from

T, B, D, H, C = 4, 5, 6, 7, 3


def sqs_net(**kwargs) -> FeedForwardSNN:
    torch.manual_seed(0)
    return FeedForwardSNN([SQSLayer(D, H), SQSLayer(H, C)], **kwargs)


def qlif_net(**kwargs) -> FeedForwardSNN:
    torch.manual_seed(0)
    return FeedForwardSNN([QLIFLayer(D, H), QLIFLayer(H, C)], **kwargs)


def rand_input(n_channels: int | None = None) -> torch.Tensor:
    torch.manual_seed(0)
    if n_channels is None:
        return (torch.rand(T, B, D) < 0.5).float()
    return (torch.rand(T, B, n_channels, D) < 0.5).float()


# --------------------------------------------------------------------- forward
def test_sqs_forward_shapes():
    out = sqs_net()(rand_input())
    assert out.spikes_out.shape == (T, B, C, 1)
    assert out.rate_out.shape == (B, C, 1)
    assert out.mem_out is None                  # no membrane: the loss uses p instead
    assert out.p_out.shape == (T, B, C, 1)
    assert out.n_spikes.shape == (T, B)
    assert len(out.n_spikes_layers) == 2
    assert out.layer_aux is None


def test_qlif_forward_shapes():
    out = qlif_net()(rand_input())
    assert out.spikes_out.shape == (T, B, C, 1)
    assert out.mem_out.shape == (T, B, C)
    assert out.p_out.shape == (T, B, C, 1)
    assert torch.equal(out.p_out[..., 0], out.mem_out)


def test_record_hidden_off_skips_stacking():
    out = sqs_net(record_hidden=False)(rand_input())
    assert out.spikes_hid == [] and out.p_hid is None and out.mem_hid is None
    assert out.n_spikes.shape == (T, B)  # counts are still there


def test_record_hidden_and_aux():
    out = qlif_net(record_hidden=True, record_aux=True)(rand_input())
    assert len(out.spikes_hid) == 1 and out.spikes_hid[0].shape == (T, B, H, 1)
    assert out.p_hid is not None and out.p_hid[0].shape == (T, B, H, 1)
    assert out.mem_hid is not None and out.mem_hid[0].shape == (T, B, H)
    assert out.layer_aux is not None and len(out.layer_aux) == 2
    assert len(out.layer_aux[0]) == T and "mem" in out.layer_aux[0][0]


def test_rate_out_is_mean_over_time_and_differentiable():
    net = sqs_net()
    out = net(rand_input())
    assert torch.allclose(out.rate_out, out.spikes_out.mean(0))
    out.rate_out.sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_n_spikes_matches_manual_count():
    net = sqs_net(record_hidden=True)
    out = net(rand_input())
    manual = out.spikes_hid[0].sum(dim=(2, 3)) + out.spikes_out.sum(dim=(2, 3))
    assert torch.allclose(out.n_spikes, manual)
    assert not out.n_spikes.requires_grad
    assert torch.allclose(torch.stack(out.n_spikes_layers).sum(0), out.n_spikes)


def test_channel_broadcast_and_mismatch():
    """A one-channel input is lifted to an N = 2 layer; any other channel count is an error."""
    torch.manual_seed(0)
    net = FeedForwardSNN([SQSLayer(D, H, 2, 1), SQSLayer(H, C, 2, 1)])
    out = net(rand_input())  # [T,B,D] -> broadcast 1 -> 2 channels
    assert out.spikes_out.shape == (T, B, C, 2)
    out = net(rand_input(2))  # already 2 channels
    assert out.spikes_out.shape == (T, B, C, 2)
    with pytest.raises(ValueError, match="only a single channel can be broadcast"):
        net(rand_input(3))


def test_wrong_in_features_raises():
    net = sqs_net()
    with pytest.raises(ValueError, match="in_features"):
        net((torch.rand(T, B, D + 1) < 0.5).float())


def test_detach_state_keeps_every_parameter_trainable():
    net = sqs_net(detach_state=True)
    out = net(rand_input())
    out.p_out.sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_units_and_param_count():
    net = sqs_net()
    assert net.units_per_layer() == [H, C]
    assert net.total_units() == H + C
    assert net.param_count() == (D * H + 3 * H) + (H * C + 3 * C)


# ---------------------------------------------------------------------- losses
def test_membrane_ce_is_sum_over_time():
    torch.manual_seed(2)
    mem = torch.randn(T, B, C)
    y = torch.randint(0, C, (B,))
    manual = sum(F.cross_entropy(mem[t], y) for t in range(T))
    assert torch.allclose(membrane_ce_loss(mem, y), manual)
    assert torch.allclose(membrane_ce_loss(mem.unsqueeze(-1), y), manual)  # [T,B,C,1] too
    with pytest.raises(ValueError):
        membrane_ce_loss(torch.randn(B, C), y)


def test_the_sqs_loss_uses_the_born_probability():
    """SQS has no membrane, so ``p`` (mean over its N channels) stands in, and that is flagged."""
    out = sqs_net()(rand_input())
    logits, substituted = logits_from(out)
    assert substituted is True
    assert torch.equal(logits, out.p_out.mean(dim=-1))

    out_q = qlif_net()(rand_input())
    logits_q, substituted_q = logits_from(out_q)
    assert substituted_q is False and logits_q is out_q.mem_out


def test_compute_loss_is_windowed_membrane_ce():
    out = qlif_net()(rand_input())
    y = torch.arange(B) % C
    loss, logits, _ = compute_loss(out, y, loss_name="membrane_ce", n_classes=C, window=2)
    assert torch.allclose(loss, membrane_ce_loss(out.mem_out, y, window=2))
    with pytest.raises(KeyError):
        compute_loss(out, y, loss_name="rate", n_classes=C)


# --------------------------------------------------------------------- metrics
def test_decoders():
    spikes = torch.zeros(T, B, C, 2)
    spikes[:, :, 1, :] = 1.0
    assert torch.equal(rate_decode(spikes), torch.full((B,), 1))
    mem = torch.zeros(T, B, C)
    mem[:, :, 2] = 5.0
    assert torch.equal(membrane_decode(mem), torch.full((B,), 2))


def test_accuracy_and_confusion():
    pred = torch.tensor([0, 1, 2, 2])
    tgt = torch.tensor([0, 1, 1, 2])
    assert accuracy(pred, tgt) == pytest.approx(0.75)
    cm = confusion_matrix(pred, tgt, 3)
    assert cm.shape == (3, 3)
    assert cm[1, 2] == 1 and cm[0, 0] == 1 and cm.sum() == 4


def test_spike_metrics():
    n = torch.full((T, B), 10.0)
    assert spikes_per_step(n) == pytest.approx(10.0)
    assert per_neuron_rate(10.0, 20) == pytest.approx(0.5)
    rates = layer_rates([torch.full((T, B), 6.0), torch.full((T, B), 4.0)], [3, 2])
    assert rates[0]["spikes_per_step"] == pytest.approx(6.0)
    assert rates[1]["per_neuron_rate"] == pytest.approx(2.0)


def test_run_metrics_accumulates_like_one_pass():
    torch.manual_seed(3)
    spikes = (torch.rand(T, 2 * B, C, 1) < 0.5).float()
    mem = torch.randn(T, 2 * B, C)
    n_spk = torch.rand(T, 2 * B) * 10
    y = torch.randint(0, C, (2 * B,))

    whole = RunMetrics(n_classes=C, units_per_layer=[H, C])
    whole.update(targets=y, spikes_out=spikes, n_spikes=n_spk, mem_out=mem, loss=1.0)

    split = RunMetrics(n_classes=C, units_per_layer=[H, C])
    for sl in (slice(0, B), slice(B, 2 * B)):
        split.update(
            targets=y[sl],
            spikes_out=spikes[:, sl],
            n_spikes=n_spk[:, sl],
            mem_out=mem[:, sl],
            loss=1.0,
        )
    assert split.acc == pytest.approx(whole.acc)
    assert split.acc_mem == pytest.approx(whole.acc_mem)
    assert split.spikes_per_step == pytest.approx(whole.spikes_per_step, rel=1e-6)
    assert split.summary()["per_neuron_rate"] == pytest.approx(whole.per_neuron_rate, rel=1e-6)


def test_run_metrics_counts_spikes_per_step_and_per_layer():
    """Every unit of both layers spiking on every step: H + C spikes per step, rate 1."""
    acc = RunMetrics(n_classes=C, units_per_layer=[H, C])
    acc.update(
        targets=torch.zeros(B, dtype=torch.long),
        spikes_out=torch.ones(T, B, C, 1),
        n_spikes=torch.full((T, B), float(H + C)),
        n_spikes_layers=[torch.full((T, B), float(H)), torch.full((T, B), float(C))],
    )
    assert acc.spikes_per_step == pytest.approx(H + C)
    assert acc.per_neuron_rate == pytest.approx(1.0)
    assert [r["per_neuron_rate"] for r in acc.layer_rates()] == pytest.approx([1.0, 1.0])
