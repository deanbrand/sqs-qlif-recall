"""Training loops. ``train_surrogate`` is the backprop-through-time trainer used
for every run."""
from sqs_bench.train.loops import EpochResult, run_epoch
from sqs_bench.train.surrogate import train_surrogate

__all__ = ["train_surrogate", "run_epoch", "EpochResult"]
