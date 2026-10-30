"""The scripts under scripts/: the shipped tables, partial tables, and the dry-run launcher."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )


def write_subset(folder: Path, keep) -> None:
    """Copy both shipped tables into ``folder``, keeping the rows ``keep(row)`` accepts."""
    folder.mkdir()
    for table in ("runs.csv", "test_history.csv"):
        with (RESULTS / table).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if keep(r)]
            fields = reader.fieldnames
        with (folder / table).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def test_report_numbers_on_the_shipped_tables():
    done = run_script("report_numbers.py")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "20 of 20 claims match, 0 differ, 0 skipped." in done.stdout


@pytest.mark.parametrize(
    ("label", "keep"),
    [
        ("seed 0", lambda r: r["seed"] == "0"),
        ("seeds 0-1", lambda r: int(r["seed"]) <= 1),
        ("seeds 0-4", lambda r: int(r["seed"]) <= 4),
        ("seeds 5-9", lambda r: int(r["seed"]) >= 5),
        # the paired models of the factorial set ran different seeds
        ("unpaired", lambda r: not (r["model"] == "sqsnn" and int(r["seed"]) % 2 == 0)
                               and not (r["model"] == "sqsnn_read" and int(r["seed"]) % 2 == 1)),
    ],
)
def test_report_numbers_skips_what_a_partial_set_cannot_support(tmp_path, label, keep):
    """A claim runs only on complete seed sets, so a subset of the shipped tables gives
    ``ok`` or skipped claims, never ``DIFF``."""
    write_subset(tmp_path / "part", keep)
    done = run_script("report_numbers.py", "--results", str(tmp_path / "part"))
    assert "Traceback" not in done.stderr, done.stderr
    assert done.returncode == 0, done.stdout
    assert "DIFF" not in done.stdout and " 0 differ, " in done.stdout
    assert "skipped" in done.stdout and "missing seeds" in done.stdout


def test_report_numbers_skips_claims_with_empty_cells(tmp_path):
    write_subset(tmp_path / "part", lambda r: True)
    # one empty cell for the loss claim and one for the QLIF levelled-off claim
    for table, column, model, variant in (("runs.csv", "test_loss", "sqsnn", "sweep60_delay1"),
                                          ("test_history.csv", "test_acc_mem", "qlif", "sweep60_delay4")):
        path = tmp_path / "part" / table
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows, fields = list(reader), reader.fieldnames
        hit = next(r for r in rows if r["model"] == model and r["variant"] == variant)
        hit[column] = ""
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    done = run_script("report_numbers.py", "--results", str(tmp_path / "part"))
    assert "Traceback" not in done.stderr, done.stderr
    assert done.returncode == 0, done.stdout
    assert "test_loss is ''" in done.stdout and "non-finite test_acc_mem" in done.stdout
    assert "18 of 20 claims match, 0 differ, 2 skipped." in done.stdout


def fake_run(outputs: Path, model: str, variant: str, seed: int, suffix: str = "", **metrics):
    """A fake run directory with just the fields ``collect_results.py`` reads."""
    run_dir = outputs / "sqs_ce" / "memprobe" / f"{model}_{variant}{suffix}" / f"seed{seed}"
    run_dir.mkdir(parents=True)
    record = {
        "config": {"dataset": {"lead_in": 1, "dedupe_test": False}},
        "model": model, "variant": variant, "delay": 1, "seed": seed, "T": 7,
        "epochs": 2, "stopped_epoch": 2, "stop_reason": "disabled",
        "test_accuracy_mem": 0.5, "test_loss": 4.0, "train_loss": 4.0,
        "n_params_trainable": 848, "device": "cpu", "train_seconds": 1.0, **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "history.csv").write_text(
        "epoch,split,loss,acc,acc_mem\n1,train,4.1,0.3,0.3\n1,test,4.0,0.5,0.5\n", encoding="utf-8")
    return run_dir


def test_collect_results_rejects_duplicate_runs(tmp_path):
    fake_run(tmp_path / "out", "sqsnn_read", "long150_delay1", 0)
    fake_run(tmp_path / "out", "sqsnn_read", "long150_delay1", 0, suffix="_again")
    done = run_script("collect_results.py", "--outputs", str(tmp_path / "out"),
                      "--results", str(tmp_path / "res"))
    assert done.returncode != 0 and "Traceback" not in done.stderr
    assert "model sqsnn_read, variant long150_delay1, seed 0" in done.stderr
    assert "_again" in done.stderr
    assert not (tmp_path / "res").exists()


def test_collect_results_leaves_out_failed_runs(tmp_path):
    out = tmp_path / "out"
    fake_run(out, "sqsnn_read", "long150_delay1", 0)
    bad = [fake_run(out, "sqsnn_read", "long150_delay1", 1, stop_reason="nonfinite", train_loss=None),
           fake_run(out, "sqsnn_read", "long150_delay1", 2, test_accuracy_mem=None),
           fake_run(out, "sqsnn_read", "long150_delay1", 3, test_loss=None)]
    done = run_script("collect_results.py", "--outputs", str(out), "--results", str(tmp_path / "res"))
    assert done.returncode == 0, done.stderr
    for run_dir in bad:
        assert f"left out the failed run {run_dir}" in done.stdout
    with (tmp_path / "res" / "runs.csv").open(newline="", encoding="utf-8") as fh:
        assert [r["seed"] for r in csv.DictReader(fh)] == ["0"]


def test_plot_recall_names_the_missing_cells(tmp_path):
    write_subset(tmp_path / "part", lambda r: not r["variant"].startswith("plateau"))
    done = run_script("plot_recall.py", "--runs", str(tmp_path / "part" / "runs.csv"),
                      "--out", str(tmp_path / "fig"))
    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "sqsnn_read plateau_delay16" in done.stderr


def test_run_experiments_dry_run_counts_every_run():
    done = run_script("run_experiments.py", "--tier", "all", "--dry-run", "--device", "cpu")
    assert done.returncode == 0, done.stderr
    counts = [int(line.rsplit("= ", 1)[1].split()[0]) for line in done.stdout.splitlines()
              if line.startswith("# ") and line.endswith(" runs")]
    assert counts == [45, 40, 20, 80, 64]
    assert done.stdout.count("OMP_NUM_THREADS=") == done.stdout.count("sqs-bench -m") == 5
