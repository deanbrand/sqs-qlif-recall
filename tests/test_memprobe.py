"""The synthetic memory probe: sequence layout, labels, seeds, dedupe and the loader."""
from __future__ import annotations

import pytest
import torch

from sqs_bench.config import resolve_cfg
from sqs_bench.data.memprobe import make_memprobe, make_memprobe_pair, memprobe_T
from sqs_bench.train.data_api import load_data


# ----------------------------------------------------------------------- layout
@pytest.mark.parametrize("lead_in", [0, 1])
@pytest.mark.parametrize("delay", [1, 2, 4, 8, 16])
def test_memprobe_T_formula(delay, lead_in):
    """``T = lead_in + 1 (cue) + delay + 1 (go) + n_readout``."""
    x, y = make_memprobe(n=32, delay=delay, n_readout=3, lead_in=lead_in, seed=1)
    assert memprobe_T(delay, 3, lead_in) == lead_in + 1 + delay + 1 + 3
    assert x.shape == (32, lead_in + 1 + delay + 1 + 3, 5)
    assert y.shape == (32,)


def test_memprobe_structure():
    delay, K, n_readout = 4, 4, 3
    x, y = make_memprobe(n=64, K=K, delay=delay, n_readout=n_readout, p_noise=0.0, seed=7)
    # step 0 is exactly the one-hot cue
    assert torch.equal(x[:, 0].argmax(dim=1), y)
    assert torch.equal(x[:, 0].sum(dim=1), torch.ones(64))
    # the go line fires at step 1 + delay, and (with no noise) nowhere else
    assert torch.equal(x[:, 1 + delay, K], torch.ones(64))
    assert float(x[:, :, K].sum()) == 64.0


@pytest.mark.parametrize("lead_in", [1, 2])
def test_lead_in_moves_the_cue_and_the_go_pulse(lead_in):
    """``lead_in`` noise-only steps come BEFORE the cue; everything after shifts with it."""
    delay, K = 3, 4
    x, y = make_memprobe(n=64, K=K, delay=delay, n_readout=3, p_noise=0.0, lead_in=lead_in,
                         seed=7)
    assert float(x[:, :lead_in].abs().sum()) == 0.0           # nothing before the cue
    assert torch.equal(x[:, lead_in].argmax(dim=1), y)        # the cue, at step lead_in
    assert torch.equal(x[:, lead_in].sum(dim=1), torch.ones(64))
    go = lead_in + 1 + delay
    assert torch.equal(x[:, go, K], torch.ones(64))
    assert float(x[:, :, K].sum()) == 64.0

    # the lead-in steps are noise like the delay, not a second silent frame
    noisy, _ = make_memprobe(n=64, K=K, delay=delay, p_noise=1.0, lead_in=lead_in, seed=7)
    assert float(noisy[:, :lead_in].min()) == 1.0
    assert torch.equal(noisy[:, lead_in].sum(dim=1), torch.ones(64))


def test_memprobe_noise_is_only_outside_the_cue_step():
    x, _ = make_memprobe(n=2000, delay=6, n_readout=3, p_noise=1.0, seed=3)
    assert float(x[:, 0].sum()) == 2000.0  # the cue step is one-hot whatever the noise
    assert float(x[:, 1:].min()) == 1.0     # everything else is on


def test_memprobe_rejects_a_bad_shape():
    for bad in ({"delay": -1}, {"K": 1}, {"n_readout": 0}, {"lead_in": -1}):
        with pytest.raises(ValueError, match="bad memprobe shape"):
            make_memprobe(n=8, seed=0, **bad)


# ----------------------------------------------------------------------- labels
def test_memprobe_labels_are_balanced():
    _, y = make_memprobe(n=8000, K=4, seed=11)
    counts = torch.bincount(y, minlength=4)
    assert counts.tolist() == [2000, 2000, 2000, 2000]
    assert not torch.equal(y, torch.arange(8000) % 4)  # ... and permuted


def test_memprobe_is_deterministic_in_the_seed():
    a = make_memprobe(n=64, seed=5)
    b = make_memprobe(n=64, seed=5)
    c = make_memprobe(n=64, seed=6)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_memprobe_pair_uses_disjoint_streams():
    """Train and test come from seeds ``seed*1000 + 1`` and ``seed*1000 + 2``."""
    train, test = make_memprobe_pair(n_train=64, n_test=64, seed=2)
    assert not torch.equal(train.tensors[0], test.tensors[0])
    again, _ = make_memprobe_pair(n_train=64, n_test=64, seed=2)
    assert torch.equal(train.tensors[0], again.tensors[0])
    assert torch.equal(train.tensors[0], make_memprobe(n=64, seed=2001)[0])
    assert torch.equal(test.tensors[0], make_memprobe(n=64, seed=2002)[0])
    assert train.info.n_features == 5 and train.info.n_classes == 4
    assert train.T == memprobe_T(8, 3) and train.D == 5


# ------------------------------------------------------------------------ dedupe
def test_dedupe_leaves_the_training_set_alone():
    """Only the test split is refilled, so a run with dedupe trains on exactly the same data."""
    plain_tr, plain_te = make_memprobe_pair(n_train=500, n_test=200, delay=1, lead_in=1, seed=4)
    dd_tr, dd_te = make_memprobe_pair(n_train=500, n_test=200, delay=1, lead_in=1, seed=4,
                                      dedupe_test=True)
    assert torch.equal(plain_tr.tensors[0], dd_tr.tensors[0])
    assert torch.equal(plain_tr.tensors[1], dd_tr.tensors[1])
    assert not torch.equal(plain_te.tensors[0], dd_te.tensors[0])
    assert dd_te.info.n_test == len(dd_te) == 200


def test_dedupe_refills_from_fresh_draws_deterministically():
    """The refill draws come from ``seed*1000 + 3, + 4, ...``, so the deduped set is
    reproducible, holds no training row, and at delay 1 really does contain refilled rows."""
    kwargs = dict(n_train=500, n_test=200, delay=1, lead_in=1, seed=4, dedupe_test=True)
    tr, te = make_memprobe_pair(**kwargs)
    _, again = make_memprobe_pair(**kwargs)
    assert torch.equal(te.tensors[0], again.tensors[0])
    assert torch.equal(te.tensors[1], again.tensors[1])

    # the cue still gives the label, so every refilled row is a valid probe row
    assert torch.equal(te.tensors[0][:, 1, :4].argmax(dim=1), te.tensors[1])

    first_test = make_memprobe(n=200, delay=1, lead_in=1, seed=4002)[0]
    first = {x.reshape(-1).numpy().tobytes() for x in first_test}
    seen = {x.reshape(-1).numpy().tobytes() for x in tr.tensors[0]}
    rows = [x.reshape(-1).numpy().tobytes() for x in te.tensors[0]]
    assert not any(r in seen for r in rows)
    assert any(r not in first for r in rows), "expected at least one refilled row at delay 1"


@pytest.mark.parametrize("sizes", [{"n_test": 9}, {"n_test": 2}, {"n_test": 0}, {"n_train": 10}])
def test_pair_sizes_must_be_positive_multiples_of_K(sizes):
    with pytest.raises(ValueError, match="positive multiple of K=4"):
        make_memprobe_pair(**{"n_train": 64, "n_test": 32, "seed": 0, **sizes})


def test_dedupe_gives_up_when_every_input_is_in_train():
    """Without noise there are only K distinct inputs, and train holds all of them."""
    with pytest.raises(ValueError, match="too few distinct inputs"):
        make_memprobe_pair(n_train=64, n_test=32, p_noise=0.0, dedupe_test=True, seed=0)


# ------------------------------------------------------------------------ loader
def test_memprobe_load_data(make_cfg):
    cfg = resolve_cfg(make_cfg(["device=cpu", "train.batch_size=8", "variant=smoke_delay8"]))
    train_loader, test_loader, info, dev = load_data(cfg, 0)
    assert info.n_features == 5 and info.n_classes == 4
    assert info.n_train == 8000 and info.n_test == 2000
    x, y = next(iter(train_loader))
    assert x.shape == (8, 13, 5) and x.dtype is torch.float32
    assert y.dtype is torch.int64
    assert any("memory probe" in d for d in dev)


def test_the_loader_carries_lead_in_and_dedupe_through(make_cfg):
    cfg = resolve_cfg(
        make_cfg(["dataset.delay=1", "dataset.lead_in=1", "dataset.dedupe_test=true",
                  "dataset.n_train=500", "dataset.n_test=200", "variant=smoke_delay1"])
    )
    _, test_loader, info, _ = load_data(cfg, 4)
    x, y = test_loader.dataset.tensors
    expected = make_memprobe_pair(n_train=500, n_test=200, delay=1, lead_in=1, seed=4,
                                  dedupe_test=True)[1]
    assert torch.equal(x, expected.tensors[0]) and torch.equal(y, expected.tensors[1])
    assert x.shape[1] == cfg.resolved.T == 1 + 1 + 1 + 1 + 3


def test_only_the_memory_probe_is_a_dataset(make_cfg):
    cfg = resolve_cfg(make_cfg(["dataset.generator=other"]))
    with pytest.raises(ValueError, match="unknown dataset"):
        load_data(cfg, 0)
