"""Gather finished runs into two tables.

Reads ``metrics.json`` and ``history.csv`` from every
``<outputs>/sqs_ce/memprobe/<model>_<tier>_delay<d>/seed<k>/`` and writes

* ``<results>/runs.csv``: one row per run (final test accuracy and losses, how training
  stopped, model size, wall time);
* ``<results>/test_history.csv``: test accuracy and loss after every epoch.

Accuracies are fractions in [0, 1], as the runs record them. ``test_accuracy_mem`` reads each
output neuron's spike probability (SQS) or membrane (QLIF). Runs whose tier is not one of
:data:`TIERS` are left out, and so are failed runs (see :func:`failure`), with a warning naming
each one.
Two usable runs with the same model, variant and seed, for example a re-run under another
``train.run_suffix``, stop the script with an error that lists them; move one of the two out of
``<outputs>`` first.

    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --outputs my_outputs --results my_results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TIERS = ("sweep60", "long150", "plateau", "factorial", "angle")

RUN_COLUMNS = [
    "tier", "model", "variant", "delay", "seed", "lead_in", "dedupe_test", "T",
    "epochs", "stopped_epoch", "stop_reason",
    "test_accuracy_mem", "test_loss", "train_loss",
    "n_params_trainable", "device", "train_seconds",
]
HISTORY_COLUMNS = ["model", "variant", "seed", "epoch", "test_acc_mem", "test_loss"]


def run_tables(metrics: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """One run's ``metrics.json`` and ``history.csv`` rows -> its rows of the two tables."""
    dataset = metrics["config"]["dataset"]  # the run's config.yaml, fully resolved
    variant = metrics["variant"]
    row = {
        "tier": variant.rsplit("_delay", 1)[0],
        "model": metrics["model"],
        "variant": variant,
        "delay": metrics["delay"],
        "seed": metrics["seed"],
        "lead_in": dataset["lead_in"],
        "dedupe_test": str(dataset["dedupe_test"]).lower(),
        "T": metrics["T"],
        "epochs": metrics["epochs"],  # the cap
        "stopped_epoch": metrics["stopped_epoch"],
        "stop_reason": metrics["stop_reason"],  # disabled (no early stopping), cap or criterion
        "test_accuracy_mem": metrics["test_accuracy_mem"],
        "test_loss": metrics["test_loss"],
        "train_loss": metrics["train_loss"],
        "n_params_trainable": metrics["n_params_trainable"],
        "device": metrics["device"].split(":")[0].split()[0],  # "cuda:0 (<gpu name>)" -> "cuda"
        "train_seconds": metrics["train_seconds"],
    }
    test = [
        {
            "model": row["model"],
            "variant": variant,
            "seed": row["seed"],
            "epoch": h["epoch"],
            "test_acc_mem": h["acc_mem"],
            "test_loss": h["loss"],
        }
        for h in history
        if h["split"] == "test"
    ]
    return row, test


def failure(metrics: dict) -> str | None:
    """Why ``metrics.json`` records a failed run, or None for a usable one.

    A run fails when its train loss became non-finite (``stop_reason`` ``nonfinite``; the
    trainer stops there) or when its final test accuracy or a final loss is missing or not
    finite (``metrics.json`` stores NaN and infinity as null).
    """
    if metrics.get("stop_reason") == "nonfinite" or metrics.get("nonfinite_loss"):
        return "the train loss became non-finite (stop_reason=nonfinite)"
    for key in ("test_accuracy_mem", "test_loss", "train_loss"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{key} is {value!r}"
    return None


def read_run(run_dir: Path) -> tuple[dict, list[dict]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as fh:
        history = list(csv.DictReader(fh))
    return metrics, history


def write_results(runs: list[tuple[dict, list[dict]]], results: Path) -> None:
    """Write both tables, ordered by tier, model, delay and seed."""
    runs = sorted(runs, key=lambda r: (TIERS.index(r[0]["tier"]), r[0]["model"], r[0]["delay"], r[0]["seed"]))
    results.mkdir(parents=True, exist_ok=True)
    with (results / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(row for row, _ in runs)
    with (results / "test_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for _, test in runs:
            writer.writerows(test)
    n_epochs = sum(len(test) for _, test in runs)
    print(f"wrote {len(runs)} runs to {results / 'runs.csv'} and {n_epochs} epochs to "
          f"{results / 'test_history.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"), help="out_root of the runs")
    parser.add_argument("--results", type=Path, default=Path("results"), help="where the tables go")
    args = parser.parse_args()

    where = args.outputs / "sqs_ce" / "memprobe"
    runs, other, dirs = [], 0, {}
    for path in sorted(where.glob("*/seed*/metrics.json")):
        metrics, history = read_run(path.parent)
        if metrics["variant"].rsplit("_delay", 1)[0] not in TIERS:
            other += 1
            continue
        reason = failure(metrics)
        if reason:
            print(f"warning: left out the failed run {path.parent}: {reason}")
            continue
        row, test = run_tables(metrics, history)
        runs.append((row, test))
        dirs.setdefault((row["model"], row["variant"], row["seed"]), []).append(path.parent)
    if other:
        print(f"left out {other} runs under {where} whose variant is not <tier>_delay<d> "
              f"with a tier in {TIERS}")
    duplicates = {key: found for key, found in dirs.items() if len(found) > 1}
    if duplicates:
        lines = [f"  model {m}, variant {v}, seed {s}: " + ", ".join(map(str, found))
                 for (m, v, s), found in sorted(duplicates.items())]
        raise SystemExit("more than one run for the same (model, variant, seed); keep one of each "
                         "under the outputs folder and collect again:\n" + "\n".join(lines))
    if not runs:
        raise SystemExit(f"nothing to write: no finished run of the five experiment sets under {where}")
    write_results(runs, args.results)


if __name__ == "__main__":
    main()
