"""Run directory: where a single (protocol, dataset, model, seed) run writes.

Layout (identical for ``sqs-bench`` and ``sqs-bench -m``)::

    <out_root>/<protocol>/<dataset>/<model><variant><run_suffix>/seed<seed>/
        config.yaml     fully resolved config actually used
        history.csv     one row per (epoch, split)
        metrics.json    final record of the run
        run.log         everything the run printed
        model.pt        only when train.save_model=true
        .hydra/         hydra's own config snapshot

The same template is written into ``hydra.run.dir`` / ``hydra.sweep.subdir`` in
``conf/config.yaml``; :func:`resolve_run_dir` prefers hydra's answer and falls back
to rebuilding the path (which is what the unit tests exercise).
"""
from __future__ import annotations

import csv
import json
import os
import math
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from omegaconf import DictConfig, OmegaConf

from sqs_bench import vtag

HISTORY_FIELDS = [
    "epoch",
    "split",
    "loss",
    "acc",
    "acc_mem",
    "spikes_per_step",
    "per_neuron_rate",
    "lr",
    "seconds",
]


def run_dir_for(
    out_root: str | Path,
    protocol: str,
    dataset: str,
    model: str,
    seed: int,
    suffix: str = "",
    variant: str = "",
) -> Path:
    """``<out_root>/<protocol>/<dataset>/<model><variant><suffix>/seed<seed>``.

    ``suffix`` is ``cfg.run_suffix``, an optional extra tag (empty by default). ``variant`` is
    ``cfg.variant`` (empty for untagged runs) and separates one run family from
    another -- ``sweep60_delay8`` becomes ``_sweep60_delay8``, exactly as
    ``${vtag:${variant}}`` does in ``conf/config.yaml``.
    """
    tag = vtag(variant)
    return (
        Path(out_root)
        / str(protocol)
        / str(dataset)
        / f"{model}{tag}{suffix}"
        / f"seed{int(seed)}"
    )


def resolve_run_dir(cfg: DictConfig) -> Path:
    """Hydra's output dir when running under hydra, else the same path rebuilt."""
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return run_dir_for(
            cfg.get("out_root", "outputs"),
            cfg.protocol.name,
            cfg.dataset.name,
            cfg.model.name,
            cfg.seed,
            str(cfg.get("run_suffix", "") or ""),
            str(cfg.get("variant", "") or ""),
        )


def is_done(run_dir: str | Path) -> bool:
    """A run is done when its ``metrics.json`` exists *and* parses with a test accuracy.

    ``write_metrics`` writes atomically, so a run killed mid-write (killed, full disk)
    leaves no file rather than a truncated one; the parse guard covers files damaged by
    other means so a rerun redoes them instead of skipping.
    """
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return "test_accuracy" in json.load(handle)
    except (OSError, ValueError):
        return False


def write_config(run_dir: str | Path, cfg: DictConfig) -> Path:
    path = Path(run_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def _sanitise(obj: Any) -> Any:
    """Make ``obj`` JSON-safe: omegaconf -> plain, tensors -> lists, NaN -> None."""
    if isinstance(obj, (DictConfig,)) or OmegaConf.is_config(obj):
        return _sanitise(OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, dict):
        return {str(k): _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):  # torch.Tensor / np.ndarray / np scalar
        return _sanitise(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_metrics(run_dir: str | Path, metrics: dict) -> Path:
    path = Path(run_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic: a kill during the write must not leave a truncated metrics.json that
    # skip_if_done would then treat as a finished run
    tmp = path.with_name(".metrics.json.tmp")
    tmp.write_text(json.dumps(_sanitise(metrics), indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryWriter:
    """Append-only ``history.csv``: the header is written once, then each row is appended
    as it arrives, so an interrupted run still leaves a valid file."""

    def __init__(self, run_dir: str | Path, fields: list[str] | None = None) -> None:
        self.path = Path(run_dir) / "history.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields or HISTORY_FIELDS)
        self._write_header()

    def _write_header(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fields).writeheader()

    def append(self, row: dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in self.fields}
        for k, v in clean.items():
            if isinstance(v, float) and math.isnan(v):
                clean[k] = ""
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fields).writerow(clean)


class _Tee:
    def __init__(self, stream, fh) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, data: str) -> int:
        self._fh.write(data)
        self._fh.flush()
        return self._stream.write(data)

    def flush(self) -> None:
        self._fh.flush()
        self._stream.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, item):  # pragma: no cover - passthrough
        return getattr(self._stream, item)


@contextmanager
def tee_stdout(run_dir: str | Path, filename: str = "run.log") -> Iterator[Path]:
    """Mirror stdout/stderr into ``run_dir/run.log`` for the duration of the block."""
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()
