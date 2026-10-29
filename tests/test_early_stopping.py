"""Early stopping: train until the train loss stops moving instead of to a fixed epoch count.

Comparing models at a fixed epoch count measures the budget as much as the neuron, so the
plateau runs train each run until none of its last 60 epochs beats the best train loss before
them by more than 1e-4 (``train.patience=60``, ``train.min_delta=1e-4``, cap 300) and record
the epoch that happened at.

The rule watches TRAIN loss. Every run reports a held-out score, and stopping on the test
curve would pick the epoch that flatters that very number.
:func:`test_monitored_quantity_is_train_loss` pins that down.
"""
from __future__ import annotations

import json

import pytest

from sqs_bench.cli import run_training
from sqs_bench.train.surrogate import MONITOR, early_stopping_cfg, should_stop
from sqs_bench.utils.rundir import run_dir_for

#: QLIF samples nothing, so the run is deterministic, and at this learning rate it fits the
#: small training set and then plateaus well before the 40-epoch cap
LEARNS = ["model=qlif", "train.lr=0.05"]
STOPPING = ["train.epochs=40", "train.patience=3", "train.min_delta=0.01"]


def train_rows(metrics: dict) -> list[dict]:
    return [r for r in metrics["history"] if r["split"] == "train"]


def saved_metrics(tmp_path, model: str = "qlif") -> dict:
    run_dir = run_dir_for(tmp_path, "sqs_ce", "memprobe", model, 0, variant="smoke_delay1")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- the rule


def test_should_stop_compares_the_last_window_with_the_best_before_it():
    """Stop when none of the last ``patience`` epochs beats the best value before them by
    more than ``min_delta``."""
    assert not should_stop([1.0, 0.9, 0.8], patience=None, min_delta=0.01)   # disabled
    assert not should_stop([1.0, 0.9, 0.8], patience=2, min_delta=0.01)      # still improving
    assert not should_stop([1.0, 0.9], patience=2, min_delta=0.01)           # too short to judge
    assert should_stop([1.0, 1.0, 1.0], patience=2, min_delta=0.01)          # two flat epochs
    # one epoch in the window that beats the best before it (1.0) is enough to go on
    assert not should_stop([1.0, 0.5, 1.0], patience=2, min_delta=0.01)
    # an improvement smaller than min_delta is not an improvement
    assert should_stop([1.0, 0.999, 0.998], patience=2, min_delta=0.01)
    assert not should_stop([1.0, 0.999, 0.998], patience=2, min_delta=0.0)


def test_should_stop_judges_the_window_against_the_best_before_it_not_a_running_best():
    """Epoch 4 (0.78) beats epoch 2 (0.89) by more than 0.1, but at epoch 5 the window is
    epochs 4-5 and the best before it is epoch 3 (0.80), which neither beats by 0.1."""
    losses = [1.0, 0.89, 0.80, 0.78, 0.77]
    stops = [should_stop(losses[:k], patience=2, min_delta=0.1) for k in range(1, len(losses) + 1)]
    assert stops == [False, False, False, False, True]  # first stops at epoch 5


def test_the_plateau_runs_rule_needs_sixty_flat_epochs():
    """``patience=60, min_delta=1e-4``: 59 flat epochs are not enough, 60 are."""
    values = [4.0, 3.0] + [3.0] * 59
    assert not should_stop(values, patience=60, min_delta=1e-4)
    assert should_stop(values + [3.0], patience=60, min_delta=1e-4)
    # a fall smaller than min_delta does not count as an improvement; a bigger one does
    assert should_stop([3.0] + [3.0 - 5e-5] * 60, patience=60, min_delta=1e-4)
    assert not should_stop([3.0] + [3.0] * 59 + [2.9998], patience=60, min_delta=1e-4)


def test_early_stopping_cfg_validates(short_cfg):
    cfg = short_cfg(["train.patience=3", "train.min_delta=0.5", "train.monitor_tail=2"])
    patience, min_delta, tail_len = early_stopping_cfg(cfg)
    assert (patience, min_delta) == (3, 0.5)
    assert tail_len == 4  # widened to patience + 1 so the decision stays auditable

    assert early_stopping_cfg(short_cfg())[0] is None  # default: disabled
    with pytest.raises(ValueError):
        early_stopping_cfg(short_cfg(["train.patience=0"]))
    with pytest.raises(ValueError):
        early_stopping_cfg(short_cfg(["train.min_delta=-1.0"]))


def test_the_plateau_settings_resolve(short_cfg):
    cfg = short_cfg(["train.epochs=300", "train.patience=60", "train.min_delta=0.0001"])
    assert early_stopping_cfg(cfg) == (60, pytest.approx(1e-4), 61)


# ---------------------------------------------------------------------- the loop


def test_patience_null_trains_the_full_budget(short_cfg):
    """The default is a fixed epoch count: no stopping at all."""
    metrics = run_training(short_cfg(["model=qlif", "train.epochs=4"]))

    assert len(train_rows(metrics)) == 4
    assert metrics["stopped_epoch"] == 4 == metrics["epochs"]
    assert metrics["stopped_early"] is False
    assert metrics["patience"] is None
    assert not any("early stopping" in d for d in metrics["deviations"])


def test_stops_early_on_a_task_it_has_already_learnt(short_cfg):
    """64 training rows and a 40-epoch cap: the fit stalls long before the cap."""
    metrics = run_training(short_cfg([*LEARNS, *STOPPING]))

    assert metrics["stopped_early"] is True
    assert metrics["stopped_epoch"] < metrics["epochs"] == 40
    assert len(train_rows(metrics)) == metrics["stopped_epoch"]
    losses = [r["loss"] for r in train_rows(metrics)]
    assert metrics["monitor_best"] < losses[0] - 0.3  # it really did learn something
    # the recorded tail must actually justify the stop
    assert should_stop(metrics["monitor_tail"], patience=3, min_delta=0.01)
    assert any("early stopping" in d for d in metrics["deviations"])


def test_patience_counts_epochs_without_improvement(short_cfg):
    """min_delta so large that nothing after epoch 1 improves -> stop at 1 + patience."""
    metrics = run_training(
        short_cfg(["model=qlif", "train.epochs=20", "train.patience=2", "train.min_delta=1e9"])
    )
    assert metrics["stopped_early"] is True
    assert metrics["stopped_epoch"] == 3


@pytest.mark.parametrize("model", ["sqsnn", "sqsnn_read"])
def test_patience_works_for_the_sampled_neurons(short_cfg, model):
    """The same rule on an SQS run, whose train loss is noisy because every spike is sampled."""
    metrics = run_training(
        short_cfg([f"model={model}", "train.epochs=20", "train.patience=2", "train.min_delta=1e9"])
    )
    assert metrics["stopped_epoch"] == 3 and metrics["stop_reason"] == "criterion"
    assert len(train_rows(metrics)) == 3


def test_monitored_quantity_is_train_loss(short_cfg):
    """Not test loss: the test set is scored once here, yet the run still stops.

    ``eval_every=1000`` leaves the forced pass on the stopping epoch as the only
    test pass, so there was no test curve to stop on. The recorded tail is exactly
    the tail of the train-loss column of ``history``.
    """
    metrics = run_training(short_cfg([*LEARNS, *STOPPING, "train.eval_every=1000"]))
    assert metrics["monitor"] == MONITOR == "train_loss"
    assert metrics["stopped_early"] is True

    losses = [r["loss"] for r in train_rows(metrics)]
    tail = metrics["monitor_tail"]
    assert len(tail) == 10                    # train.monitor_tail default
    assert tail == pytest.approx(losses[-len(tail):])

    test_rows = [r for r in metrics["history"] if r["split"] == "test"]
    assert len(test_rows) == 1                # only the forced eval on the stopping epoch
    assert test_rows[0]["epoch"] == metrics["stopped_epoch"]
    assert metrics["test_accuracy"] is not None
    assert tail[-1] != pytest.approx(test_rows[0]["loss"])


def test_metrics_json_carries_the_stopping_fields(short_cfg, tmp_path):
    run_training(short_cfg([*LEARNS, *STOPPING]))
    saved = saved_metrics(tmp_path)
    for key in (
        "stopped_epoch",
        "stopped_early",
        "monitor",
        "patience",
        "min_delta",
        "monitor_tail",
    ):
        assert key in saved, key
    assert saved["patience"] == 3 and saved["min_delta"] == pytest.approx(0.01)
    assert saved["epochs"] == 40 and saved["stopped_epoch"] < 40
    assert saved["monitor"] == "train_loss"
