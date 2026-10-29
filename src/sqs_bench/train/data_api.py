"""Data for a run: build the dataset named by the config and wrap it in loaders.

``dataset.generator=memprobe`` generates the synthetic memory probe in-process
(:mod:`sqs_bench.data.memprobe`); it is the only dataset. Both splits carry the same
:class:`~sqs_bench.data.memprobe.DatasetInfo` (feature and class counts, split sizes).
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from sqs_bench.data.memprobe import DatasetInfo, make_memprobe_pair


def _memprobe(cfg_ds: Any, seed: int) -> tuple[Any, Any]:
    return make_memprobe_pair(
        n_train=int(cfg_ds.get("n_train", 8000)),
        n_test=int(cfg_ds.get("n_test", 2000)),
        K=int(cfg_ds.get("K", 4)),
        delay=int(cfg_ds.get("delay", 8)),
        n_readout=int(cfg_ds.get("n_readout", 3)),
        p_noise=float(cfg_ds.get("p_noise", 0.1)),
        lead_in=int(cfg_ds.get("lead_in", 0)),
        dedupe_test=bool(cfg_ds.get("dedupe_test", False)),
        seed=int(seed),
        name=str(cfg_ds.name),
    )


def make_loaders(
    train: TensorDataset,
    test: TensorDataset,
    batch_size: int,
    *,
    seed: int,
    drop_last_train: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Seeded train/test loaders. Train is shuffled, test never is.

    The train loader shuffles with its own generator; the test loader has none, so
    iterating it draws once per epoch from the global CPU torch RNG.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def load_data(cfg: Any, seed: int) -> tuple[DataLoader, DataLoader, DatasetInfo, list[str]]:
    """Return ``(train_loader, test_loader, info, deviations)``."""
    cfg_ds = cfg.dataset
    name = str(cfg_ds.name)
    batch_size = int(cfg.train.batch_size)
    num_workers = int(cfg.train.get("num_workers", 0))
    drop_last = bool(cfg.train.get("drop_last_train", False))
    dev: list[str] = []

    if str(cfg_ds.get("generator", "") or "") == "memprobe":
        train_ds, test_ds = _memprobe(cfg_ds, seed)
        dev.append(
            f"dataset {name}: synthetic memory probe generated in-process "
            f"(delay={int(cfg_ds.get('delay', 8))}, p_noise={float(cfg_ds.get('p_noise', 0.1))}, "
            f"seeds {int(seed) * 1000 + 1}/{int(seed) * 1000 + 2})"
        )
    else:
        raise ValueError(f"unknown dataset {name!r}: set dataset.generator=memprobe")

    info = train_ds.info

    train_loader, test_loader = make_loaders(
        train_ds,
        test_ds,
        batch_size,
        seed=int(seed),
        drop_last_train=drop_last,
        num_workers=num_workers,
    )
    return train_loader, test_loader, info, dev
