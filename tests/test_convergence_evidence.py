"""Convergence has to be auditable, and failure states must not read as results.

**The plateau rule is not convergence.** ``should_stop`` watches a noisy online average
of the train loss while the optimiser is still moving the model. It fires on a genuine
plateau, on a bad basin, on vanishing temporal credit, and on noise, and it cannot tell
them apart. So the record carries the slope of the monitored tail with its standard
error, where the best epoch fell, whether the run reached its cap, and the tail itself --
and these tests pin that a reader can recompute every one of them from ``metrics.json``
alone.

**A silent network scores chance.** ``argmax`` over an all-zero score vector returns
class 0, so an output layer that has stopped spiking reports a respectable-looking
number on a balanced task. The end-to-end test below builds exactly that network -- a
model with every weight zeroed -- and shows the record refusing to let it pass:
``silent_frac`` 1.0, no decided windows at all, and a train loss sitting on the exact
uninformed value ``n_steps * ln(C)``.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from sqs_bench.cli import run_training
from sqs_bench.config import resolve_cfg, resolved_hidden
from sqs_bench.losses import membrane_ce_loss
from sqs_bench.metrics import (
    SILENT_FRAC_ALERT,
    UNIFORM_LOSS_RTOL,
    RunMetrics,
    collect_readout,
    decision_health,
    uniform_output_loss,
)
from sqs_bench.models.build import build_model
from sqs_bench.train.data_api import load_data
from sqs_bench.train.surrogate import (
    STILL_IMPROVING_T,
    STOP_REASONS,
    provenance,
    should_stop,
    tail_stats,
    train_surrogate,
)
from sqs_bench.utils.rundir import run_dir_for

#: the same deterministic QLIF run the early-stopping tests use: it learns, then plateaus
LEARNS = ["model=qlif", "train.lr=0.05"]
STOPPING = ["train.epochs=40", "train.patience=3", "train.min_delta=0.01"]


def saved_metrics(tmp_path, model: str = "qlif") -> dict:
    run_dir = run_dir_for(tmp_path, "sqs_ce", "memprobe", model, 0, variant="smoke_delay1")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def broken_model_run(short_cfg, tmp_path, model: str, fill: float, overrides: list[str]) -> dict:
    """Train a network whose every parameter is pinned to ``fill``.

    ``train.lr=0`` keeps it pinned: Adam's step is ``-lr * ...``, so the degenerate
    network is still degenerate when the final pass scores it. ``fill=0`` gives a
    completely silent output layer; ``fill=nan`` gives a non-finite loss.
    """
    cfg = resolve_cfg(short_cfg([f"model={model}", "train.lr=0.0", *overrides]))
    train_loader, test_loader, info, _ = load_data(cfg, int(cfg.seed))
    model = build_model(
        cfg.model,
        in_features=int(info.n_features),
        n_classes=int(info.n_classes),
        hidden=resolved_hidden(cfg),
        detach_state=bool(cfg.resolved.detach_state),
    )
    for p in model.parameters():
        p.data.fill_(fill)
    return train_surrogate(
        cfg, model, (train_loader, test_loader), info, torch.device("cpu"),
        run_dir=tmp_path / "broken",
    )


# ------------------------------------------------------------------ the tail statistic


def test_tail_stats_describes_a_flat_tail():
    stats = tail_stats([0.5] * 12)
    assert stats["tail_window"] == 12
    assert stats["tail_slope"] == pytest.approx(0.0)
    assert stats["tail_drop"] == pytest.approx(0.0)
    assert stats["tail_resid_std"] == pytest.approx(0.0)
    assert stats["still_improving"] is False


def test_tail_stats_recovers_a_known_slope():
    values = [1.0 - 0.01 * i for i in range(10)]
    stats = tail_stats(values)
    assert stats["tail_slope"] == pytest.approx(-0.01)
    assert stats["tail_drop"] == pytest.approx(0.09)
    # relative to a mean level of 0.955: about -1 % of the loss per epoch
    assert stats["tail_rel_slope"] == pytest.approx(-0.01 / 0.955, rel=1e-6)
    assert stats["still_improving"] is True


def test_tail_stats_calls_a_noisy_flat_tail_flat():
    """Alternating noise around a constant: no trend, so nothing to wait for."""
    values = [0.5 + 0.02 * (-1) ** i for i in range(20)]
    stats = tail_stats(values)
    assert abs(stats["tail_slope_t"]) < abs(STILL_IMPROVING_T)
    assert stats["still_improving"] is False
    assert stats["tail_resid_std"] > 0.01          # the noise really is there


def test_the_slope_catches_a_descent_the_plateau_rule_calls_a_stall():
    """The whole point: ``stopped_early`` and "converged" are different claims.

    ``min_delta`` large enough swallows a steady descent, so the plateau rule fires
    while the loss is still falling by half a percent an epoch. Someone reading only
    ``stopped_early`` would call that converged; ``still_improving`` says otherwise.
    """
    values = [1.0 - 0.005 * i + 0.0005 * (-1) ** i for i in range(20)]
    assert should_stop(values, patience=5, min_delta=0.03) is True

    stats = tail_stats(values)
    assert stats["tail_slope"] == pytest.approx(-0.005, abs=1e-4)
    assert stats["tail_slope_t"] < STILL_IMPROVING_T
    assert stats["still_improving"] is True


def test_tail_stats_refuses_to_guess():
    short = tail_stats([1.0, 0.9])
    assert short["tail_window"] == 2
    assert short["tail_drop"] == pytest.approx(0.1)
    assert short["tail_slope"] is None and short["still_improving"] is None

    assert tail_stats([])["tail_drop"] is None
    nan = tail_stats([1.0, float("nan"), 0.5, 0.4])
    assert nan["tail_slope"] is None and nan["still_improving"] is None


# --------------------------------------------------------- convergence in metrics.json


def test_a_capped_run_says_so(short_cfg):
    """No patience: the cap always ends the run, and the record must not imply more."""
    metrics = run_training(short_cfg(["model=qlif", "train.epochs=4"]))
    assert metrics["stop_reason"] == "disabled"
    assert metrics["hit_epoch_cap"] is True
    assert metrics["criterion_met"] is False
    assert metrics["stopped_early"] is False
    assert metrics["stopped_epoch"] == metrics["epochs"] == 4


def test_a_run_the_rule_stopped_says_so(short_cfg):
    metrics = run_training(short_cfg([*LEARNS, *STOPPING]))
    assert metrics["stop_reason"] == "criterion"
    assert metrics["criterion_met"] is True
    assert metrics["stopped_early"] is True
    assert metrics["hit_epoch_cap"] is False
    assert metrics["stopped_epoch"] < 40
    assert metrics["stop_reason"] in STOP_REASONS


def test_the_rule_firing_on_the_cap_epoch_is_not_an_early_stop(short_cfg):
    """``min_delta=1e9`` stops at ``1 + patience``; set the cap to exactly that epoch.

    No epoch was saved and the result is every bit as budget-limited as a plain cap hit.
    ``criterion_met`` keeps the fact that the rule fired; ``stopped_early`` and
    ``hit_epoch_cap`` keep the fact that it bought nothing.
    """
    metrics = run_training(
        short_cfg(["model=qlif", "train.epochs=3", "train.patience=2", "train.min_delta=1e9"])
    )
    assert metrics["stopped_epoch"] == metrics["epochs"] == 3
    assert metrics["criterion_met"] is True
    assert metrics["stopped_early"] is False
    assert metrics["hit_epoch_cap"] is True
    assert metrics["stop_reason"] == "cap"
    assert any("right-censored" in d for d in metrics["deviations"])


def test_the_tail_statistics_can_be_recomputed_from_the_record(short_cfg, tmp_path):
    """Every ``tail_*`` field is a function of ``monitor_tail``, which is also stored.

    That is the standard: a reader judges the stop from the artefact, not from trust.
    """
    run_training(short_cfg([*LEARNS, *STOPPING]))
    saved = saved_metrics(tmp_path)
    recomputed = tail_stats(saved["monitor_tail"])
    for key, value in recomputed.items():
        if value is None:
            assert saved[key] is None, key
        elif isinstance(value, bool):
            assert saved[key] is value, key
        else:
            assert saved[key] == pytest.approx(value), key


def test_the_best_epoch_and_the_distance_to_it_are_recorded(short_cfg):
    metrics = run_training(short_cfg([*LEARNS, *STOPPING]))
    train_losses = [r["loss"] for r in metrics["history"] if r["split"] == "train"]
    assert metrics["monitor_best"] == pytest.approx(min(train_losses))
    assert metrics["monitor_best_epoch"] == train_losses.index(min(train_losses)) + 1
    assert metrics["monitor_last"] == pytest.approx(train_losses[-1])
    assert metrics["epochs_since_best"] == metrics["stopped_epoch"] - metrics["monitor_best_epoch"]


def test_a_healthy_run_raises_no_flags(short_cfg):
    """The control. Without it, every flag below could be firing on everything.

    ``sqsnn_read_uni`` starts at U(0, pi), so its output layer spikes from the first epoch.
    """
    metrics = run_training(short_cfg(["model=sqsnn_read_uni", *STOPPING]))
    assert metrics["failure_flags"] == []
    assert metrics["readout_silent"] is False
    assert metrics["at_uniform_loss"] is False
    assert metrics["nonfinite_loss"] is False
    assert metrics["single_class_predictions"] is False
    assert metrics["silent_frac"] < SILENT_FRAC_ALERT
    assert metrics["accuracy_decided"] is not None


# ------------------------------------------------------------------ silence and ties


def test_decision_health_separates_silence_from_a_tie():
    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],   # silent -> also tied
            [2.0, 2.0, 1.0, 0.0],   # tied on a non-zero score
            [3.0, 1.0, 0.0, 0.0],   # decided
        ]
    )
    silent, tied = decision_health(scores)
    assert silent.tolist() == [True, False, False]
    assert tied.tolist() == [True, True, False]
    with pytest.raises(ValueError):
        decision_health(torch.zeros(2, 3, 4))


def test_a_silent_readout_scores_chance_and_is_counted_as_silent():
    """Four balanced classes, no output spikes at all: 25 % on the page, 0 % of it real."""
    acc = RunMetrics(n_classes=4)
    targets = torch.arange(4).repeat(8)                 # exactly balanced
    acc.update(targets=targets, spikes_out=torch.zeros(3, 32, 4))

    assert acc.acc == pytest.approx(0.25)               # looks exactly like chance
    health = acc.readout_health()
    assert health["silent_frac"] == 1.0
    assert health["tie_frac"] == 1.0
    assert health["n_decided"] == 0
    assert health["accuracy_decided"] is None           # nothing was ever decided
    assert health["readout_silent"] is True
    assert health["single_class_predictions"] is True
    assert health["pred_class_fraction"] == [1.0, 0.0, 0.0, 0.0]


def test_a_tie_without_silence_is_still_a_tie():
    acc = RunMetrics(n_classes=4)
    scores = torch.ones(3, 6, 4) * 2.0                  # every class equally loud
    acc.update(targets=torch.zeros(6, dtype=torch.long), spikes_out=scores)
    health = acc.readout_health()
    assert health["silent_frac"] == 0.0                 # not silent...
    assert health["tie_frac"] == 1.0                    # ...but decided by index order
    assert health["readout_silent"] is False


def test_the_silence_alert_needs_more_than_half_the_windows():
    def silent_fraction(n_silent: int, n_total: int) -> dict:
        acc = RunMetrics(n_classes=4)
        spikes = torch.zeros(2, n_total, 4)
        spikes[:, n_silent:, 1] = 1.0                   # the rest vote for class 1
        acc.update(targets=torch.ones(n_total, dtype=torch.long), spikes_out=spikes)
        return acc.readout_health()

    half = silent_fraction(5, 10)
    assert half["silent_frac"] == pytest.approx(SILENT_FRAC_ALERT)
    assert half["readout_silent"] is False              # the threshold is strict
    assert half["accuracy_decided"] == pytest.approx(1.0)   # the decided half was right
    assert half["n_decided"] == 5

    over = silent_fraction(6, 10)
    assert over["silent_frac"] == pytest.approx(0.6)
    assert over["readout_silent"] is True


def test_accuracy_decided_ignores_the_tie_broken_windows():
    """A network right on every window it actually decided, silent on the rest.

    Plain accuracy hides that behind the tie-break; ``accuracy_decided`` does not.
    """
    acc = RunMetrics(n_classes=4)
    spikes = torch.zeros(2, 8, 4)
    spikes[:, :2, 3] = 1.0                              # two windows vote class 3
    acc.update(targets=torch.full((8,), 3, dtype=torch.long), spikes_out=spikes)
    health = acc.readout_health()
    assert acc.acc == pytest.approx(2 / 8)
    assert health["accuracy_decided"] == pytest.approx(1.0)
    assert health["n_decided"] == 2


def test_the_membrane_readout_is_audited_too():
    acc = RunMetrics(n_classes=3)
    acc.update(
        targets=torch.zeros(4, dtype=torch.long),
        spikes_out=torch.ones(2, 4, 3),
        mem_out=torch.zeros(2, 4, 3),
    )
    health = acc.readout_health()
    assert health["silent_frac_mem"] == 1.0
    assert health["tie_frac_mem"] == 1.0


def test_collect_readout_hands_the_accumulator_back():
    """``run_epoch`` keeps its accumulator private; this is how the counters get out."""
    with collect_readout() as bucket:
        inner = RunMetrics(n_classes=2)
    outside = RunMetrics(n_classes=2)
    assert len(bucket) == 1 and bucket[0] is inner       # identity: the dataclasses compare equal
    assert all(item is not outside for item in bucket)   # nothing is captured after the block


# --------------------------------------------------------------- the uninformed loss


def test_uniform_loss_matches_the_objective_it_claims_to_describe():
    """Not a remembered constant: the same number the loss function itself returns."""
    logits = torch.full((3, 16, 4), 2.5)                # equal across classes
    y = torch.arange(4).repeat(4)
    assert uniform_output_loss("membrane_ce", 4, 3) == pytest.approx(3 * math.log(4))
    assert uniform_output_loss("membrane_ce", 4, 3) == pytest.approx(4.158883083, abs=1e-8)
    assert float(membrane_ce_loss(logits, y)) == pytest.approx(uniform_output_loss("membrane_ce", 4, 3))


def test_uniform_loss_declines_to_answer_when_it_cannot():
    assert uniform_output_loss("some_other_loss", 4, 3) is None
    assert uniform_output_loss("membrane_ce", 1, 3) is None
    assert uniform_output_loss("membrane_ce", 4, 0) is None


# ------------------------------------------------------- the failure states, end to end


@pytest.mark.parametrize("model", ["sqsnn", "qlif"])
def test_a_silent_network_cannot_pass_as_chance(short_cfg, tmp_path, model):
    """A network that has stopped spiking, reproduced and caught.

    Every weight zeroed: the output layer emits nothing, so ``argmax`` returns class 0
    on every window and the membrane readout is a flat zero vector. The reported
    accuracy is then simply the share of class 0 in the test set -- a number that would
    read as "chance" on a balanced task -- while the loss sits on ``n_steps * ln(C)``, the
    exact value of an output that says nothing. ``n_steps`` is the response window (3),
    not ``T``.
    """
    metrics = broken_model_run(short_cfg, tmp_path, model, 0.0, ["train.epochs=2"])

    assert metrics["silent_frac"] == 1.0
    assert metrics["tie_frac"] == 1.0
    assert metrics["silent_frac_mem"] == 1.0
    assert metrics["n_decided"] == 0
    assert metrics["accuracy_decided"] is None
    assert metrics["readout_silent"] is True
    assert metrics["single_class_predictions"] is True
    assert metrics["pred_class_fraction"][0] == 1.0

    n_classes, steps = 4, int(metrics["response_window"])
    assert steps == 3 < metrics["T"]
    assert metrics["uniform_loss"] == pytest.approx(steps * math.log(n_classes))
    assert metrics["train_uniform_loss_gap"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["test_uniform_loss_gap"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["at_uniform_loss"] is True

    assert set(metrics["failure_flags"]) == {
        "silent_readout",
        "single_class_predictions",
        "at_uniform_loss",
    }
    assert any("failure states detected" in d for d in metrics["deviations"])
    # and the number it would otherwise have reported, unlabelled, is still there
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


def test_at_uniform_loss_uses_the_declared_relative_tolerance(short_cfg, tmp_path):
    metrics = broken_model_run(short_cfg, tmp_path, "sqsnn", 0.0, ["train.epochs=1"])
    gap, uniform = metrics["train_uniform_loss_gap"], metrics["uniform_loss"]
    assert abs(gap) <= UNIFORM_LOSS_RTOL * uniform
    assert metrics["train_loss"] == pytest.approx(uniform + gap)


@pytest.mark.parametrize("model", ["sqsnn", "qlif"])
def test_a_non_finite_loss_stops_the_run_instead_of_eating_the_budget(short_cfg, tmp_path, model):
    """NaN loses every comparison, so the plateau rule can never fire on it.

    Without the guard the run would spend its whole cap and then serialise as ``null``.
    """
    metrics = broken_model_run(
        short_cfg, tmp_path, model, float("nan"), ["train.epochs=30", "train.patience=3"]
    )
    assert metrics["nonfinite_loss"] is True
    assert metrics["stop_reason"] == "nonfinite"
    assert metrics["stopped_epoch"] == 1 < metrics["epochs"]
    assert metrics["criterion_met"] is False
    assert "nonfinite_loss" in metrics["failure_flags"]
    assert not math.isfinite(metrics["train_loss"])
    assert metrics["train_uniform_loss_gap"] is None      # no gap to a number that is not one


# ---------------------------------------------------------------------- provenance

PROVENANCE_KEYS = {
    "gpu_name",
    "gpu_capability",
    "gpu_memory_mb",
    "gpu_count",
    "python_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "n_params_stored",
    "n_params_trainable",
}


def test_provenance_records_the_card_and_the_libraries():
    """The GPU as its own field (``device`` holds it only as a display string), the library
    versions and the determinism flags -- and nothing about the machine beyond that."""
    model = torch.nn.Linear(4, 3)
    prov = provenance(torch.device("cpu"), model)
    assert set(prov) == PROVENANCE_KEYS
    assert prov["gpu_name"] is None and prov["gpu_count"] == 0   # this run is on the CPU
    assert prov["deterministic_algorithms"] is False            # recorded, not assumed
    assert prov["torch_version"] == torch.__version__
    assert prov["n_params_stored"] == prov["n_params_trainable"] == 15


def test_provenance_separates_stored_from_trainable_parameters():
    """What makes a frozen-gate arm auditable from its own record."""
    model = torch.nn.Linear(4, 3)
    model.bias.requires_grad_(False)
    prov = provenance(torch.device("cpu"), model)
    assert prov["n_params_stored"] == 15
    assert prov["n_params_trainable"] == 12


def test_metrics_json_carries_the_provenance(short_cfg, tmp_path):
    run_training(short_cfg(["model=sqsnn_angle_pi4", "train.epochs=2"]))
    saved = saved_metrics(tmp_path, "sqsnn_angle_pi4")
    assert saved["gpu_name"] is None                # CPU run
    assert set(saved["provenance"]) == PROVENANCE_KEYS
    assert saved["n_params_trainable"] == saved["provenance"]["n_params_trainable"] == 780
    assert saved["provenance"]["n_params_stored"] == 848   # the frozen read gate is stored
