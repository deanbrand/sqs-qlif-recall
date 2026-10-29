"""A synthetic memory probe with a controllable delay.

The network must *hold* a cue across a delay before it can answer:

======  ===========================================================
step    what the ``K + 1`` input lines carry
======  ===========================================================
<lead>  ``lead_in`` steps of Bernoulli(``p_noise``) noise (default 0)
0       a one-hot **cue** on line ``y``; every other line silent
1..d    Bernoulli(``p_noise``) on every line -- the **delay**
1+d     the **go** line (index ``K``) is forced to 1, the rest noise
rest    Bernoulli(``p_noise``) -- ``n_readout`` steps to answer in
======  ===========================================================

so ``T = lead_in + 1 + delay + 1 + n_readout`` and the label is the cue line
(step numbers above count from the cue). Sweeping ``delay`` measures memory span
directly: a memoryless network is at chance, and a network whose state decays
over ``k`` steps drops to chance once ``delay > k``.

The loss and both readouts see only the last ``response_window`` steps
(``= n_readout``, set in ``conf/dataset/memprobe.yaml``), never the cue frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

#: Fresh test draws ``make_memprobe_pair`` makes with ``dedupe_test`` before it gives up.
MAX_REFILLS = 100


@dataclass(frozen=True)
class DatasetInfo:
    """What a run needs to know about the data: sizes of the input, the output and the splits."""

    name: str
    n_features: int
    n_classes: int
    n_train: int
    n_test: int


def memprobe_T(delay: int, n_readout: int, lead_in: int = 0) -> int:
    """``lead_in + 1 + delay + 1 + n_readout`` -- lead-in, cue, delay, go, readout."""
    return int(lead_in) + 1 + int(delay) + 1 + int(n_readout)


def make_memprobe(
    *,
    n: int,
    K: int = 4,
    delay: int = 8,
    n_readout: int = 3,
    p_noise: float = 0.1,
    lead_in: int = 0,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """``(x [n, T, K+1] float32, y [n] int64)``, deterministic in ``seed``.

    Labels are exactly balanced over the ``K`` cue lines (when ``K`` divides ``n``)
    and then permuted, so no split of the data is class-skewed.
    """
    n, K, delay, n_readout = int(n), int(K), int(delay), int(n_readout)
    lead_in = int(lead_in)
    if n < 1 or K < 2 or delay < 0 or n_readout < 1 or lead_in < 0:
        raise ValueError(
            f"bad memprobe shape: n={n}, K={K}, delay={delay}, "
            f"n_readout={n_readout}, lead_in={lead_in}")
    T = memprobe_T(delay, n_readout, lead_in)
    D = K + 1  # K cue lines + one go line

    g = torch.Generator().manual_seed(int(seed))
    x = (torch.rand((n, T, D), generator=g) < float(p_noise)).to(torch.float32)
    y = (torch.arange(n) % K)[torch.randperm(n, generator=g)].to(torch.int64)

    # lead_in: noise-only steps BEFORE the cue, so the cue need not be the first input the
    # network sees.  sweep60 uses 0; long150, plateau, factorial and angle use 1.
    x[:, lead_in, :] = 0.0                # the cue step carries the cue and nothing else
    x[torch.arange(n), lead_in, y] = 1.0
    x[:, lead_in + 1 + delay, K] = 1.0    # the go pulse, on top of that step's noise
    return x, y


class MemProbeDataset(TensorDataset):
    """``(x [T, K+1] float32, y int64)`` pairs carrying a :class:`DatasetInfo`."""

    def __init__(self, x: Tensor, y: Tensor, info: DatasetInfo) -> None:
        super().__init__(x, y)
        self.info = info
        self.T = int(x.shape[1])
        self.D = int(x.shape[2])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MemProbeDataset(T={self.T}, D={self.D}, n={len(self)})"


def make_memprobe_pair(
    *,
    n_train: int = 8000,
    n_test: int = 2000,
    K: int = 4,
    delay: int = 8,
    n_readout: int = 3,
    p_noise: float = 0.1,
    lead_in: int = 0,
    dedupe_test: bool = False,
    seed: int = 0,
    name: str = "memprobe",
) -> tuple[MemProbeDataset, MemProbeDataset]:
    """Train/test split from two disjoint RNG streams (``seed*1000 + 1`` and ``+ 2``).

    ``dedupe_test`` drops test rows whose input appears verbatim in train.  The noise is Bernoulli
    over only ~29-34 free cells at short delays, so the input support is small and the two splits
    collide.  Measured at ``lead_in=1`` with 8000/2000:

        delay 1: 38.0% of test rows are also in train  (a pure train lookup scores 53.5%)
        delay 2: 24.3%                                 (43.2%)
        delay 4:  6.9%                                 (30.1%)
        delay 8:  0.4%                                 (25.3%)

    The one-hot cue makes ``x`` determine ``y``, so these are exact leaked pairs, and at delay 1-2
    the lookup ceiling is close to the accuracies our models reach.  It cannot explain a model
    *comparison* -- every model sees the same duplicated rows and the memory-blind ones sit at
    chance regardless -- but absolute accuracies at short delay are not clean generalisation.

    Default False; the plateau and angle runs set it True.  At delay >= 4 the collision is
    negligible.

    ``n_train`` and ``n_test`` must be positive multiples of ``K`` so that every cue appears
    equally often.  The refill gives up with a ``ValueError`` after :data:`MAX_REFILLS` fresh
    draws, which only happens when the input support is too small to hold ``n_test`` sequences
    that are not in train (for example ``p_noise=0``).
    """
    for split, size in (("n_train", n_train), ("n_test", n_test)):
        if int(K) < 1 or int(size) < 1 or int(size) % int(K):
            raise ValueError(
                f"{split}={size} must be a positive multiple of K={K}, "
                "so that every cue appears equally often")
    kwargs = {"K": K, "delay": delay, "n_readout": n_readout, "p_noise": p_noise,
              "lead_in": lead_in}
    x_tr, y_tr = make_memprobe(n=n_train, seed=int(seed) * 1000 + 1, **kwargs)
    x_te, y_te = make_memprobe(n=n_test, seed=int(seed) * 1000 + 2, **kwargs)
    if dedupe_test:
        # Drop-and-refill, not drop.  Plain dropping shrinks the test set (2000 -> 1239 at delay 1)
        # AND unbalances it (319/299/315/306), which moves chance from exactly 25.00% to 25.75%,
        # so a result compared against chance would be off by 0.75 pp.  Holding
        # n_test fixed also keeps evaluation consuming the same amount of randomness as an
        # un-deduped run, so training trajectories stay comparable across the flag.
        seen = {x.reshape(-1).numpy().tobytes() for x in x_tr}
        per_class = int(n_test) // int(K)
        kept: dict[int, list] = {k: [] for k in range(int(K))}
        stream = int(seed) * 1000 + 2
        refills = 0
        while any(len(v) < per_class for v in kept.values()):
            for x, y in zip(x_te, y_te):
                c = int(y)
                if len(kept[c]) < per_class and x.reshape(-1).numpy().tobytes() not in seen:
                    kept[c].append(x)
            if any(len(v) < per_class for v in kept.values()):
                if refills == MAX_REFILLS:
                    found = [len(kept[c]) for c in range(int(K))]
                    raise ValueError(
                        f"dedupe_test: after {MAX_REFILLS} fresh draws of {n_test} sequences, "
                        f"only {found} unseen test sequences per cue, {per_class} needed; "
                        "there are too few distinct inputs for this n_train, n_test, delay "
                        "and p_noise")
                refills += 1
                stream += 1                       # a fresh, disjoint draw; never reuses train
                x_te, y_te = make_memprobe(n=int(n_test), seed=stream, **kwargs)
        x_te = torch.stack([x for c in range(int(K)) for x in kept[c]])
        y_te = torch.tensor([c for c in range(int(K)) for _ in kept[c]], dtype=torch.int64)
        perm = torch.randperm(len(y_te), generator=torch.Generator().manual_seed(stream))
        x_te, y_te = x_te[perm], y_te[perm]
    info = DatasetInfo(
        name=str(name),
        n_features=int(K) + 1,
        n_classes=int(K),
        n_train=int(n_train),
        n_test=int(x_te.shape[0]),
    )
    return MemProbeDataset(x_tr, y_tr, info), MemProbeDataset(x_te, y_te, info)
