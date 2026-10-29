"""``sqs-bench`` -- one (protocol, dataset, model, seed) training run per invocation.

    uv run sqs-bench                                   # defaults: sqs_ce / memprobe / sqsnn / seed 0
    uv run sqs-bench model=sqsnn_read dataset.delay=4 variant=d4 seed=3
    uv run sqs-bench -m model=qlif,sqsnn seed=0,1,2    # a multirun
    uv run sqs-bench --cfg job                         # print the composed config, run nothing

Everything lands in ``<out_root>/<protocol>/<dataset>/<model><variant>/seed<seed>/``
(see :mod:`sqs_bench.utils.rundir`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from sqs_bench.config import resolve_cfg, resolved_hidden
from sqs_bench.models.build import build_model
from sqs_bench.train.data_api import load_data
from sqs_bench.train.surrogate import train_surrogate
from sqs_bench.utils.device import resolve_device
from sqs_bench.utils.gitinfo import git_info
from sqs_bench.utils.rundir import (
    is_done,
    resolve_run_dir,
    tee_stdout,
    write_config,
    write_metrics,
)
from sqs_bench.utils.seeding import seed_everything

CONFIG_PATH = "conf"


def run_training(cfg: DictConfig, run_dir: str | Path | None = None) -> dict:
    """Resolve the config, build everything, train, write the run directory."""
    cfg = resolve_cfg(cfg)
    name = str(cfg.train.get("name", "surrogate"))
    if name not in ("surrogate", ""):
        raise KeyError(f"unknown train.name={name!r}; use surrogate")
    run_dir = Path(run_dir) if run_dir is not None else resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("skip_if_done", False)) and is_done(run_dir):
        print(f"skip: {run_dir} already has metrics.json")
        return {"skipped": True, "run_dir": str(run_dir)}

    with tee_stdout(run_dir):
        seed_everything(int(cfg.seed))
        device = resolve_device(cfg.get("device", "auto"))

        train_loader, test_loader, info, dev_data = load_data(cfg, int(cfg.seed))
        cfg.deviations = list(cfg.get("deviations", []) or []) + list(dev_data)

        dropped: list[str] = []
        # hidden activations are only stacked on request; spike *counts* are always recorded
        record_hidden = bool(cfg.get("record_hidden", False))
        model = build_model(
            cfg.model,
            in_features=int(info.n_features),
            n_classes=int(info.n_classes),
            hidden=resolved_hidden(cfg),
            detach_state=bool(cfg.resolved.detach_state),
            record_hidden=record_hidden,
            record_aux=bool(cfg.get("record_aux", False)),
            dropped=dropped,
        ).to(device)
        cfg.deviations = list(cfg.deviations) + dropped

        write_config(run_dir, cfg)
        metrics = train_surrogate(
            cfg, model, (train_loader, test_loader), info, device, run_dir=run_dir
        )
        metrics["git"] = git_info()
        metrics["run_dir"] = str(run_dir)
        metrics["config"] = OmegaConf.to_container(cfg, resolve=True)
        write_metrics(run_dir, metrics)
        print(f"wrote {run_dir / 'metrics.json'}")
    return metrics


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> Any:
    metrics = run_training(cfg)
    # returned so hydra sweepers can optimise against it
    return metrics.get("test_accuracy")


if __name__ == "__main__":  # pragma: no cover
    main()
