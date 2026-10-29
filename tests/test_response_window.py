"""The memory probe must actually require memory.

If every time step were scored, the cue frame included, a model with no memory at all could
classify the cue frame directly and score far above chance, and the task would stop measuring
memory. The loss and both decoders therefore see only the frames *after* the go pulse, which by
construction carry no label information. These tests pin that, from the metric functions up to
a full training run.
"""
from __future__ import annotations

import collections
import math

import pytest
import torch

from sqs_bench.cli import run_training
from sqs_bench.data.encoders import identity
from sqs_bench.data.memprobe import make_memprobe, make_memprobe_pair
from sqs_bench.losses import membrane_ce_loss
from sqs_bench.metrics import RunMetrics, membrane_decode, rate_decode, window_slice
from sqs_bench.models.feedforward import FeedForwardSNN
from sqs_bench.neurons.qlif import QLIFLayer
from sqs_bench.train.loops import run_epoch


def test_window_slice_takes_the_last_steps():
    seq = torch.arange(10).reshape(10, 1, 1).float()
    assert window_slice(seq, None).shape[0] == 10
    assert window_slice(seq, 3).flatten().tolist() == [7.0, 8.0, 9.0]
    with pytest.raises(ValueError, match="exceeds"):
        window_slice(seq, 11)


def test_decoders_ignore_everything_before_the_window():
    # class 0 shouted in the early frames, class 1 in the last two
    seq = torch.zeros(6, 4, 3)
    seq[:4, :, 0] = 5.0
    seq[4:, :, 1] = 1.0
    assert rate_decode(seq).tolist() == [0] * 4           # unmasked: the early frames win
    assert rate_decode(seq, window=2).tolist() == [1] * 4  # masked: only the response window
    assert membrane_decode(seq, window=2).tolist() == [1] * 4


def test_ce_loss_uses_only_the_window():
    torch.manual_seed(0)
    mem = torch.randn(6, 8, 4)
    y = torch.randint(0, 4, (8,))
    assert torch.allclose(membrane_ce_loss(mem, y, window=6), membrane_ce_loss(mem, y))
    windowed = membrane_ce_loss(mem, y, window=2)
    assert torch.allclose(windowed, membrane_ce_loss(mem[-2:], y))
    assert not torch.allclose(windowed, membrane_ce_loss(mem, y))


def test_run_metrics_honour_the_window():
    seq = torch.zeros(5, 6, 3)
    seq[:3, :, 2] = 9.0        # a decoy before the window
    seq[3:, :, 0] = 1.0
    y = torch.zeros(6, dtype=torch.long)
    plain, masked = RunMetrics(n_classes=3), RunMetrics(n_classes=3, window=2)
    for m in (plain, masked):
        m.update(targets=y, spikes_out=seq)
    assert plain.acc == 0.0 and masked.acc == 1.0


def test_run_epoch_hands_the_window_to_the_loss():
    """The training loop passes the window on: its loss is the windowed CE of the network.

    QLIF samples nothing, so the same forward pass can be repeated outside the loop.
    """
    torch.manual_seed(0)
    model = FeedForwardSNN([QLIFLayer(5, 8), QLIFLayer(8, 4)])
    x, y = make_memprobe(n=16, delay=1, lead_in=1, seed=0)
    T = int(x.shape[1])

    def encode(batch):
        return identity(batch, T)

    with torch.no_grad():
        res, _ = run_epoch(
            model, [(x, y)], encoder=encode, device=torch.device("cpu"),
            loss_name="membrane_ce", n_classes=4, split="test", window=3,
        )
        mem = model(encode(x)).mem_out
    assert res.loss == pytest.approx(float(membrane_ce_loss(mem, y, window=3)))
    assert res.loss != pytest.approx(float(membrane_ce_loss(mem, y)))
    assert res.acc_mem == pytest.approx(float((membrane_decode(mem, window=3) == y).float().mean()))


def test_a_training_run_scores_only_the_response_window(short_cfg):
    """End to end: the memprobe config sets the window, and the record shows it was used."""
    metrics = run_training(short_cfg(["model=sqsnn"]))
    assert metrics["response_window"] == 3 < metrics["T"]
    assert metrics["config"]["resolved"]["response_window"] == 3
    # the uninformed loss is counted over the 3 scored steps, not over all T
    assert metrics["uniform_loss"] == pytest.approx(3 * math.log(4))


def test_response_window_frames_carry_no_label_information():
    """The heart of it: after the go pulse every line is Bernoulli(p_noise) regardless of
    the cue, so ANY model that cannot remember is exactly at chance on the masked metric."""
    x, y = make_memprobe(n=8000, K=4, delay=4, n_readout=3, p_noise=0.1, seed=1)
    window = x[:, -3:, :]                                  # [n, 3, K+1]
    per_class = torch.stack([window[y == c].mean(dim=(0, 1)) for c in range(4)])
    # every class induces the same mean activity per line, to within sampling error
    spread = (per_class.max(dim=0).values - per_class.min(dim=0).values).max().item()
    assert spread < 0.02, f"response window leaks the label (spread {spread:.4f})"

    # and the strongest stateless rule -- vote for the line with most activity in the
    # window -- is at chance
    pred = window.sum(dim=1)[:, :4].argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    assert 0.20 < acc < 0.30, f"a stateless voter scores {acc:.3f} on the response window"


def test_cue_frame_alone_would_have_leaked_the_label():
    """Control for the test above: without the window, the cue frame alone gives the label."""
    x, y = make_memprobe(n=4000, K=4, delay=4, n_readout=3, p_noise=0.1, seed=2)
    pred = x[:, 0, :4].argmax(dim=-1)          # just read the cue frame
    assert (pred == y).float().mean().item() > 0.99


def test_memprobe_test_set_overlaps_train_at_short_delay():
    """The probe's input support is small, so train and test collide -- badly at short delay.

    Bernoulli(0.1) noise over ~29-34 free cells means many sequences repeat.  The one-hot cue makes
    x determine y, so a duplicated row is an exact leaked (x, y) pair.  Measured at lead_in=1 with
    8000/2000: 38.0% of test rows are in train at delay 1, falling to 0.4% by delay 8, and a pure
    train-lookup table would score 53.5% at delay 1 against 25.3% at delay 8.

    This does not affect any model COMPARISON -- every model sees the same duplicates and the
    memory-blind ones score 25% regardless -- but it caps how much an absolute short-delay accuracy
    is worth.  `dedupe_test=True` removes it.  Pinned so the caveat cannot be forgotten.
    """

    def overlap(delay, **kw):
        tr, te = make_memprobe_pair(n_train=2000, n_test=500, delay=delay, lead_in=1, seed=0, **kw)
        seen = {x.reshape(-1).numpy().tobytes() for x in tr.tensors[0]}
        return sum(1 for x in te.tensors[0] if x.reshape(-1).numpy().tobytes() in seen) / len(te)

    assert overlap(1) > 0.10, "expected substantial train/test collision at delay 1"
    assert overlap(8) < 0.05, "expected the collision to vanish by delay 8"
    assert overlap(1, dedupe_test=True) == 0.0, "dedupe_test must remove every duplicate"


def test_dedupe_test_keeps_the_test_set_balanced_and_full_size():
    """Dropping duplicates alone would move chance off 25% and shrink the test set.

    Plain dropping gives 1239 rows at delay 1 with class counts 319/299/315/306, so the majority
    class is 25.75% -- an anchor stated as "must sit at chance" would be wrong by 0.75 pp, and the
    smaller eval set would also consume less randomness, perturbing the training trajectory
    relative to a run without dedupe.  `dedupe_test` therefore refills from fresh disjoint draws.
    """
    for delay in (1, 2, 8):
        tr, te = make_memprobe_pair(n_train=2000, n_test=400, delay=delay, lead_in=1, seed=0,
                                    dedupe_test=True)
        counts = collections.Counter(int(y) for y in te.tensors[1])
        assert len(te) == 400, (delay, len(te))
        assert set(counts.values()) == {100}, (delay, dict(counts))
        seen = {x.reshape(-1).numpy().tobytes() for x in tr.tensors[0]}
        assert not any(x.reshape(-1).numpy().tobytes() in seen for x in te.tensors[0]), delay
