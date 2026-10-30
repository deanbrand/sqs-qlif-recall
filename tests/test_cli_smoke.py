"""End-to-end smoke test: compose the real hydra config, train every model for one short
epoch on the memory probe, and check the run directory it leaves behind."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sqs_bench.cli import run_training
from sqs_bench.config import resolve_cfg
from sqs_bench.utils.rundir import run_dir_for

#: model -> (n_params, n_params_stored) on the 5 -> 64 -> 4 memprobe network
MODELS = {
    "lif": (576, 576),
    "qlif": (1288, 1288),
    "sqsnn": (780, 780),
    "sqsnn_uni": (780, 780),
    "sqsnn_read": (848, 848),
    "sqsnn_read_uni": (848, 848),
    "sqsnn_angle0": (780, 848),
    "sqsnn_angle_pi4": (780, 848),
    "sqsnn_angle_pi2": (780, 848),
    "sqsnn_angle_pi": (780, 848),
}

#: every key ``metrics.json`` holds; a new key must be added here
METRICS_KEYS = {
    "run_id", "finished_at", "protocol", "dataset", "model", "variant", "seed", "trainer",
    "n_params", "n_params_per_layer", "n_params_trainable", "hidden", "N", "N_mem", "T",
    "batch_size", "lr", "epochs", "stopped_epoch", "stopped_early", "criterion_met",
    "hit_epoch_cap", "stop_reason", "monitor", "patience", "min_delta", "monitor_tail",
    "monitor_best", "monitor_best_epoch", "monitor_last", "epochs_since_best", "tail_window",
    "tail_slope", "tail_slope_se", "tail_slope_t", "tail_rel_slope", "tail_drop",
    "tail_resid_std", "still_improving", "loss", "encoder", "response_window", "delay",
    "detach_state", "n_neurons", "n_units", "test_accuracy", "test_accuracy_mem",
    "best_test_accuracy", "test_loss", "train_accuracy", "train_loss", "n_scored",
    "silent_frac", "tie_frac", "n_decided", "accuracy_decided", "silent_frac_mem",
    "tie_frac_mem", "pred_class_fraction", "readout_silent", "single_class_predictions",
    "uniform_loss", "train_uniform_loss_gap", "test_uniform_loss_gap", "at_uniform_loss",
    "nonfinite_loss", "failure_flags", "spikes_per_step", "per_neuron_rate", "layer_rates",
    "train_seconds", "torch_version", "device", "gpu_name", "provenance", "n_train", "n_test",
    "deviations", "history", "git", "run_dir", "config",
}


def short_run_dir(tmp_path: Path, model: str, seed: int = 0) -> Path:
    return run_dir_for(tmp_path, "sqs_ce", "memprobe", model, seed, variant="smoke_delay1")


def test_config_composes_and_resolves(make_cfg):
    """The defaults are the study's setting: sqs_ce on memprobe at delay 8 with sqsnn."""
    cfg = resolve_cfg(make_cfg())
    assert (cfg.resolved.protocol, cfg.resolved.dataset, cfg.resolved.model) == (
        "sqs_ce", "memprobe", "sqsnn"
    )
    assert cfg.train.name == "surrogate"
    assert cfg.resolved.T == 0 + 1 + 8 + 1 + 3
    assert cfg.resolved.encoder == "identity"
    assert cfg.resolved.loss == "membrane_ce"
    assert cfg.resolved.lr == pytest.approx(5e-4)
    assert cfg.resolved.batch_size == 128
    assert cfg.resolved.hidden == 64           # memprobe's protocol_overrides
    assert cfg.resolved.epochs == 15           # the protocol default; every run sets its own
    assert cfg.resolved.response_window == 3
    assert cfg.resolved.detach_state is False  # full BPTT
    assert cfg.seed == 0 and cfg.device == "auto"


@pytest.mark.parametrize("variant", ["", "sweep60_delay8"])
def test_hydra_run_dir_matches_the_fallback(make_cfg, tmp_path, variant):
    """``hydra.run.dir`` and :func:`run_dir_for` must not drift apart."""
    cfg = make_cfg([f"variant={variant}"], hydra_cfg=True)
    expected = run_dir_for(tmp_path, "sqs_ce", "memprobe", "sqsnn", 0, variant=variant)
    assert Path(str(cfg.hydra.run.dir)) == expected
    assert Path(str(cfg.hydra.sweep.dir)) / str(cfg.hydra.sweep.subdir) == expected
    if variant:
        assert expected == tmp_path / "sqs_ce" / "memprobe" / "sqsnn_sweep60_delay8" / "seed0"
    else:
        assert expected == tmp_path / "sqs_ce" / "memprobe" / "sqsnn" / "seed0"


def test_train_one_epoch_writes_run_dir(short_cfg, tmp_path):
    metrics = run_training(short_cfg())
    run_dir = short_run_dir(tmp_path, "sqsnn")

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "run.log").exists()
    assert not (run_dir / "model.pt").exists()  # save_model defaults to false

    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(saved) == METRICS_KEYS
    assert set(saved["git"]) == {"commit", "dirty", "branch"}
    assert saved["protocol"] == "sqs_ce" and saved["model"] == "sqsnn"
    assert saved["dataset"] == "memprobe" and saved["variant"] == "smoke_delay1"
    assert saved["trainer"] == "surrogate"
    assert saved["N"] == 1 and saved["N_mem"] == 1 and saved["hidden"] == 64
    assert 0.0 <= saved["test_accuracy"] <= 1.0
    assert saved["spikes_per_step"] >= 0.0
    assert metrics["test_accuracy"] == pytest.approx(saved["test_accuracy"])
    assert saved["n_train"] == 64 and saved["n_test"] == 32

    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["split"] for r in rows] == ["train", "test"]
    assert all(r["epoch"] == "1" for r in rows)
    assert float(rows[-1]["spikes_per_step"]) >= 0.0
    assert float(rows[0]["loss"]) == pytest.approx(saved["train_loss"])
    assert float(rows[1]["acc"]) == pytest.approx(saved["test_accuracy"])


@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_model_trains_one_epoch(short_cfg, tmp_path, model):
    """Each shipped model config builds, trains one epoch and records itself correctly."""
    metrics = run_training(short_cfg([f"model={model}"]))
    n_params, n_stored = MODELS[model]
    assert metrics["model"] == model
    assert metrics["n_params"] == metrics["n_params_trainable"] == n_params
    assert metrics["provenance"]["n_params_stored"] == n_stored
    assert metrics["T"] == 1 + 1 + 1 + 1 + 3       # lead_in + cue + delay + go + readout
    assert metrics["delay"] == 1
    assert metrics["encoder"] == "identity"
    assert metrics["response_window"] == 3
    assert metrics["loss"] == "membrane_ce"
    assert metrics["lr"] == pytest.approx(5e-4)
    assert metrics["test_accuracy"] is not None and 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["test_accuracy_mem"] is not None    # both readouts are reported
    assert metrics["stop_reason"] == "disabled"
    assert (short_run_dir(tmp_path, model) / "metrics.json").exists()
    # QLIF has a membrane; the SQS output uses its Born probability in its place
    substituted = any("Born spiking probability" in d for d in metrics["deviations"])
    assert substituted == model.startswith("sqsnn")


def test_skip_if_done(short_cfg, capsys):
    run_training(short_cfg())
    out = run_training(short_cfg(["skip_if_done=true"]))
    assert out.get("skipped") is True
    assert "skip" in capsys.readouterr().out


def test_save_model(short_cfg, tmp_path):
    run_training(short_cfg(["train.save_model=true"]))
    assert (short_run_dir(tmp_path, "sqsnn") / "model.pt").exists()


def test_truncated_metrics_json_is_not_done(tmp_path):
    """``skip_if_done`` must redo a run whose ``metrics.json`` was cut short."""
    from sqs_bench.utils.rundir import is_done, write_metrics

    run = tmp_path / "run"
    assert not is_done(run)
    run.mkdir()
    (run / "metrics.json").write_text('{"test_accuracy": 0.9', encoding="utf-8")  # truncated
    assert not is_done(run)
    write_metrics(run, {"test_accuracy": 0.9})
    assert is_done(run)
    assert not (run / ".metrics.json.tmp").exists()


def test_only_the_surrogate_trainer_exists(short_cfg):
    with pytest.raises(KeyError, match="train.name"):
        run_training(short_cfg(["train.name=other"]))


def test_the_console_entry_point_trains_a_run(tmp_path):
    """The hydra entry point itself, in a fresh interpreter, with a tagged variant."""
    out_root = tmp_path / "outputs"
    cmd = [
        sys.executable, "-m", "sqs_bench.cli",
        f"out_root={out_root.as_posix()}",
        "model=sqsnn_read",
        "device=cpu",
        "dataset.delay=1",
        "dataset.lead_in=1",
        "dataset.n_train=32",
        "dataset.n_test=16",
        "train.batch_size=16",
        "train.epochs=1",
        "seed=2",
        "variant=long150_delay1",
    ]
    done = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    run_dir = out_root / "sqs_ce" / "memprobe" / "sqsnn_read_long150_delay1" / "seed2"
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["model"] == "sqsnn_read" and saved["seed"] == 2
    assert saved["variant"] == "long150_delay1"
    assert (run_dir / "history.csv").exists() and (run_dir / ".hydra").is_dir()
    assert not any("variant is empty" in d for d in saved["deviations"])
