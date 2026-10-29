"""sqs_bench -- training harness for the SQS and QLIF recall study.

Importing the package registers two small OmegaConf resolvers, so any config in
``conf/`` can use them:

``${vtag:${variant}}``
    ``""`` -> ``""``, ``"sweep60_delay8"`` -> ``"_sweep60_delay8"``. Splices the
    variant tag into the run directory without leaving a stray underscore on untagged runs.
``${sum:2,${dataset.delay},${dataset.n_readout}}``
    integer sum; used by ``conf/dataset/memprobe.yaml`` to derive ``T`` from the delay.

Registration is idempotent (``replace=True``), so re-importing under pytest or a
hydra multirun is harmless.
"""
from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

__all__ = ["vtag", "int_sum", "register_resolvers"]


def vtag(variant: Any) -> str:
    """``""`` -> ``""``; anything else gets a leading underscore (``"a"`` -> ``"_a"``)."""
    text = "" if variant is None else str(variant)
    return f"_{text}" if text else ""


def int_sum(*values: Any) -> int:
    """Integer sum -- ``${sum:2,${dataset.delay},${dataset.n_readout}}``."""
    return int(sum(int(v) for v in values))


def register_resolvers() -> None:
    """Register ``vtag`` / ``sum`` with OmegaConf (idempotent)."""
    OmegaConf.register_new_resolver("vtag", vtag, replace=True)
    OmegaConf.register_new_resolver("sum", int_sum, replace=True)


register_resolvers()
