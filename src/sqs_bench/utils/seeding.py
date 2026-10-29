"""Seeding helpers.

Every run seeds python, numpy and torch (CPU + CUDA) from ``cfg.seed``.

Which RNG each random step uses:

* the memory-probe data and the train-loader shuffle use their own seeded
  :class:`torch.Generator` objects, so they do not depend on anything else;
* parameter initialisation, the SQS measurement sampling (``torch.rand`` with no
  generator, on the compute device) and the test-loader iteration (one draw per
  epoch on the CPU) all use the **global** torch RNG, so they are interleaved and
  the order of operations matters for exact reproduction.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python / numpy / torch. Returns the seed for convenience."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

