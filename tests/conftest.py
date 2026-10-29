"""Shared fixtures: the packaged hydra config, composed for short CPU runs."""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import sqs_bench

CONF_DIR = str(Path(sqs_bench.__file__).parent / "conf")

#: a memory-probe run small enough to train an epoch in well under a second on a CPU
SHORT_RUN = (
    "device=cpu",
    "dataset.delay=1",
    "dataset.lead_in=1",
    "dataset.n_train=64",
    "dataset.n_test=32",
    "train.batch_size=16",
    "train.epochs=1",
    "variant=smoke_delay1",
)


def compose_cfg(overrides=(), *, hydra_cfg: bool = False):
    """``conf/config.yaml`` composed with ``overrides``, as the CLI would see it."""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        return compose(
            config_name="config", overrides=list(overrides), return_hydra_config=hydra_cfg
        )


@pytest.fixture
def make_cfg(tmp_path: Path):
    """``make_cfg(overrides)``: the composed config, writing under this test's ``tmp_path``."""

    def _make(overrides=(), *, hydra_cfg: bool = False):
        return compose_cfg([f"out_root={tmp_path.as_posix()}", *overrides], hydra_cfg=hydra_cfg)

    return _make


@pytest.fixture
def short_cfg(make_cfg):
    """``short_cfg(overrides)``: :data:`SHORT_RUN` plus ``overrides`` (later ones win)."""

    def _make(overrides=(), *, hydra_cfg: bool = False):
        return make_cfg([*SHORT_RUN, *overrides], hydra_cfg=hydra_cfg)

    return _make
