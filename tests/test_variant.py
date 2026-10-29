"""The variant mechanism: resolvers, run directories and ``derive_variant``.

A *variant* tags a run family (``sweep60_delay8``, ``plateau_delay16``, ...) so that it gets
its own run directory. The empty variant must leave the plain path untouched, and a memprobe
run without a tag must say so in its deviations.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from sqs_bench import int_sum, register_resolvers, vtag
from sqs_bench.config import derive_variant, resolve_cfg
from sqs_bench.utils.rundir import resolve_run_dir, run_dir_for


# --------------------------------------------------------------------- resolvers
@pytest.mark.parametrize(
    ("variant", "expected"),
    [("", ""), ("sweep60_delay8", "_sweep60_delay8"), ("angle_delay1", "_angle_delay1"),
     (None, "")],
)
def test_vtag(variant, expected):
    assert vtag(variant) == expected


def test_int_sum():
    assert int_sum(2, 8, 3) == 13
    assert int_sum(2, 8, 3, 1) == 14
    assert int_sum() == 0


def test_resolvers_are_registered_and_idempotent():
    register_resolvers()  # must not raise on a second registration
    cfg = OmegaConf.create(
        {"v": "long150_delay4", "tag": "${vtag:${v}}", "delay": 8, "lead_in": 1,
         "T": "${sum:2,${delay},3,${lead_in}}"}
    )
    assert cfg.tag == "_long150_delay4"
    assert cfg.T == 14


# ------------------------------------------------------------------ run directory
def test_run_dir_without_variant_is_unchanged(tmp_path):
    expected = tmp_path / "sqs_ce" / "memprobe" / "sqsnn" / "seed3"
    assert run_dir_for(tmp_path, "sqs_ce", "memprobe", "sqsnn", 3) == expected
    assert run_dir_for(tmp_path, "sqs_ce", "memprobe", "sqsnn", 3, "", "") == expected


def test_run_dir_with_variant(tmp_path):
    assert run_dir_for(tmp_path, "sqs_ce", "memprobe", "sqsnn_read", 4,
                       variant="plateau_delay16") == (
        tmp_path / "sqs_ce" / "memprobe" / "sqsnn_read_plateau_delay16" / "seed4"
    )
    # a run suffix comes after the variant
    assert run_dir_for(tmp_path, "sqs_ce", "memprobe", "qlif", 1, "_x", "sweep60_delay2") == (
        tmp_path / "sqs_ce" / "memprobe" / "qlif_sweep60_delay2_x" / "seed1"
    )


def test_resolve_run_dir_outside_hydra_rebuilds_the_path(make_cfg, tmp_path):
    cfg = make_cfg(["model=sqsnn_angle_pi2", "seed=5", "variant=angle_delay2"])
    assert resolve_run_dir(cfg) == (
        tmp_path / "sqs_ce" / "memprobe" / "sqsnn_angle_pi2_angle_delay2" / "seed5"
    )


# ---------------------------------------------------------------- derive_variant
@pytest.mark.parametrize("delay", [1, 2, 4, 8, 16])
def test_derive_variant_names_the_delay(make_cfg, delay):
    cfg = resolve_cfg(make_cfg([f"dataset.delay={delay}", f"variant=sweep60_delay{delay}"]))
    assert derive_variant(cfg) == f"delay{delay}"


def test_derive_variant_ignores_the_lead_in(make_cfg):
    """``lead_in`` moves T, but through ``dataset.T``: that is the task, not a variant."""
    cfg = resolve_cfg(make_cfg(["dataset.lead_in=1", "variant=long150_delay8"]))
    assert cfg.resolved.T == 1 + 1 + 8 + 1 + 3
    assert derive_variant(cfg) == "delay8"


def test_an_untagged_run_records_a_deviation(make_cfg):
    cfg = resolve_cfg(make_cfg())
    assert cfg.resolved.variant == ""
    assert any("variant is empty but delay8" in d for d in cfg.deviations)


def test_a_tagged_run_records_no_deviation(make_cfg):
    cfg = resolve_cfg(make_cfg(["variant=sweep60_delay8"]))
    assert cfg.resolved.variant == "sweep60_delay8"
    assert not [d for d in cfg.deviations if "variant is empty" in d]


def test_strict_variant_raises(make_cfg):
    with pytest.raises(ValueError, match="variant is empty"):
        resolve_cfg(make_cfg(["strict_variant=true"]))
    resolve_cfg(make_cfg(["strict_variant=true", "variant=sweep60_delay8"]))  # tagged: fine
