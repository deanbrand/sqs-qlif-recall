"""The identity encoder: the memory probe is already a spike train, so only the axes move."""
from __future__ import annotations

import pytest
import torch

from sqs_bench.config import resolve_cfg
from sqs_bench.data.encoders import identity
from sqs_bench.train.encoding import BatchEncoder, get_encoder
from sqs_bench.train.surrogate import make_encoder


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_identity_moves_the_time_axis():
    x = torch.rand(4, 7, 5, generator=gen())
    s = identity(x, 7)
    assert s.shape == (7, 4, 5)
    assert torch.equal(s, x.permute(1, 0, 2))
    assert s.is_contiguous()


def test_identity_rejects_the_wrong_length():
    with pytest.raises(ValueError, match="time steps"):
        identity(torch.rand(4, 7, 5), 6)


def test_identity_rejects_a_flat_batch():
    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        identity(torch.rand(4, 7), 7)


def test_get_encoder():
    fn, deviation = get_encoder("identity")
    assert fn is identity and deviation is None
    with pytest.raises(KeyError):
        get_encoder("bernoulli")


def test_batch_encoder_is_identity():
    enc = BatchEncoder("identity", 7)
    x = torch.rand(4, 7, 5, generator=gen())
    assert torch.equal(enc(x), identity(x, 7))
    assert enc.deviations == []


def test_make_encoder_follows_the_resolved_config(make_cfg):
    cfg = resolve_cfg(make_cfg(["variant=x_delay8"]))
    enc = make_encoder(cfg)
    assert enc.name == "identity" and enc.deviations == []
    assert enc.T == cfg.resolved.T == 13
