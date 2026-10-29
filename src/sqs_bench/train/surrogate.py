"""Surrogate-gradient training (SQS paper, arXiv:2506.21324, Appendix A): plain BPTT with Adam.

This is the training loop used for every run; it writes ``history.csv`` and returns
the ``metrics.json`` payload.

``train.patience`` turns on early stopping, so a run can be trained until its train
loss stops moving instead of to a fixed epoch count (see :func:`should_stop`). It is
off by default.

Three things this loop records that a bare accuracy would hide.

**Convergence is evidence, not a flag.** ``should_stop`` fires on a noisy online
train-loss plateau, which is compatible with convergence, a bad basin, vanishing
temporal credit, or plain noise. So the record carries the slope of the monitored tail
with its standard error, the best epoch and how long before the end it fell, whether
the run reached its cap, and the tail itself -- enough for a reader to judge the stop
instead of trusting it (:func:`tail_stats`).

**Failure states must not read as results.** A silent output layer scores exactly
chance on a balanced probe, and a fit sitting on the uninformed loss has learnt
nothing; both are counted and flagged
(:meth:`sqs_bench.metrics.RunMetrics.readout_health`,
:func:`sqs_bench.metrics.uniform_output_loss`). A non-finite train loss stops the run
instead of consuming the whole cap with NaN comparisons that can never trip the
plateau rule.

**Hardware is provenance.** Float reduction order and the sampling RNG depend on the
device, so the GPU, the library versions and the determinism flags are recorded per
run (:func:`provenance`).
"""
from __future__ import annotations

import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from sqs_bench.config import resolved_hidden
from sqs_bench.metrics import (
    UNIFORM_LOSS_RTOL,
    collect_readout,
    per_neuron_rate,
    uniform_output_loss,
)
from sqs_bench.models.feedforward import FeedForwardSNN
from sqs_bench.train.encoding import BatchEncoder
from sqs_bench.train.loops import run_epoch
from sqs_bench.utils.device import describe_device
from sqs_bench.utils.rundir import HistoryWriter, resolve_run_dir, utcnow


def build_optimizer(cfg: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.train.get("optimizer", "adam")).lower()
    lr = float(cfg.train.lr)
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        betas = tuple(cfg.train.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(
            params, lr=lr, betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(cfg.train.get("weight_decay", 0.0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.train.get("weight_decay", 0.0)))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.train.get("momentum", 0.0)))
    raise KeyError(f"unknown optimizer {name!r}")


#: what early stopping watches. TRAIN loss, deliberately, never test loss.
#: Runs are scored on a held-out probe; stopping on the test curve would choose
#: the epoch that happens to look best on the very number we report, which is selecting
#: on the endpoint. Train loss is a property of the fit alone, so "this run converged"
#: stays a statement about optimisation rather than about the score.
MONITOR = "train_loss"

#: How negative the tail slope has to be, in standard errors, before the run counts as
#: still descending. Two is deliberately easy to trip: a training curve's residuals are
#: autocorrelated, so this is a noise scale rather than a test with a real level. Read
#: it one way only -- a run that does NOT trip it has a genuinely flat tail; a run that
#: does had not finished descending, whatever the plateau rule said.
STILL_IMPROVING_T = -2.0

#: ``stop_reason`` values. ``disabled`` -- no patience was set, so the cap was always
#: going to end the run. ``criterion`` -- the plateau rule fired with epochs to spare.
#: ``cap`` -- the run reached ``train.epochs``, including the boundary case where the
#: rule first fired on the cap epoch and therefore saved nothing. ``nonfinite`` -- the
#: monitored loss stopped being a number.
STOP_REASONS = ("disabled", "criterion", "cap", "nonfinite")


def early_stopping_cfg(cfg: Any) -> tuple[int | None, float, int]:
    """``(patience, min_delta, tail_len)`` from ``cfg.train``; patience None = disabled."""
    patience = cfg.train.get("patience", None)
    patience = None if patience is None else int(patience)
    if patience is not None and patience < 1:
        raise ValueError(f"train.patience must be null or >= 1, got {patience}")
    min_delta = float(cfg.train.get("min_delta", 1e-4) or 0.0)
    if min_delta < 0.0:
        raise ValueError(f"train.min_delta must be >= 0, got {min_delta}")
    tail_len = int(cfg.train.get("monitor_tail", 10) or 10)
    # always keep enough history to audit the decision that stopped the run
    return patience, min_delta, max(tail_len, (patience or 0) + 1)


def should_stop(values: Sequence[float], patience: int | None, min_delta: float) -> bool:
    """True once none of the last ``patience`` epochs improves on the earlier best.

    The earlier best is the lowest value before those ``patience`` epochs. The run stops
    when none of them is lower than it by more than ``min_delta``. The epochs inside the
    window are each compared with that earlier best, not with each other: with
    ``patience=2`` and ``min_delta=0.1``, ``[1, 0.89, 0.80, 0.78, 0.77]`` stops at epoch 5,
    because 0.78 and 0.77 are within 0.1 of 0.80, even though 0.78 is more than 0.1 below
    0.89.

    This is a *plateau* rule on a noisy online average, not a convergence test. What it
    means is decided by :func:`tail_stats`, not by the fact that it fired.
    """
    if not patience or len(values) <= patience:
        return False
    best_before = min(values[: -patience])
    return all(best_before - v <= min_delta for v in values[-patience:])


def tail_stats(values: Sequence[float]) -> dict:
    """Least-squares description of the tail of the monitored trajectory.

    ``values`` is exactly the ``monitor_tail`` written to ``metrics.json``, so every
    number here can be recomputed from the record alone.

    * ``tail_slope`` -- change in the monitor per epoch over the window. Negative means
      the loss is still going down.
    * ``tail_slope_se`` / ``tail_slope_t`` -- the ordinary-least-squares standard error
      of that slope and the ratio to it, so the slope can be read against the noise
      instead of in the abstract.
    * ``tail_rel_slope`` -- the same slope divided by the mean level, which makes it
      comparable across losses of different size (3*ln(4) = 4.16 at chance for the
      memory probe).
    * ``tail_drop`` -- how far the monitor fell across the whole window.
    * ``tail_resid_std`` -- the spread about the fitted line: the epoch-to-epoch noise
      the plateau rule was fighting.
    * ``still_improving`` -- ``tail_slope_t <= STILL_IMPROVING_T``. A run that stops with
      this True stopped for some reason other than having finished.
    """
    ys = [float(v) for v in values]
    n = len(ys)
    out: dict[str, Any] = {
        "tail_window": n,
        "tail_slope": None,
        "tail_slope_se": None,
        "tail_slope_t": None,
        "tail_rel_slope": None,
        "tail_drop": (ys[0] - ys[-1]) if n >= 2 else None,
        "tail_resid_std": None,
        "still_improving": None,
    }
    if n < 3 or not all(math.isfinite(y) for y in ys):
        return out

    xs = list(range(n))
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    rss = sum((y - (mean_y + slope * (x - mean_x))) ** 2 for x, y in zip(xs, ys))
    resid_var = rss / (n - 2)
    se = math.sqrt(resid_var / sxx)
    level = sum(abs(y) for y in ys) / n

    out["tail_slope"] = slope
    out["tail_slope_se"] = se
    out["tail_resid_std"] = math.sqrt(resid_var)
    out["tail_rel_slope"] = (slope / level) if level > 0 else None
    # se == 0 means the window is an exact straight line: the slope then has no
    # sampling noise under this model, so a t ratio is not defined and the sign decides
    out["tail_slope_t"] = (slope / se) if se > 0 else None
    out["still_improving"] = bool(slope / se <= STILL_IMPROVING_T) if se > 0 else bool(slope < 0.0)
    return out


def _cudnn_version() -> int | None:
    """cuDNN version, or ``None``; a CUDA build of torch with no visible GPU raises here."""
    try:
        return torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except (RuntimeError, ValueError):
        return None


def provenance(device: torch.device, model: torch.nn.Module) -> dict:
    """What ran this, and on which GPU.

    The GPU name, capability and memory as their own fields (``device`` holds them only
    as a display string), the Python / torch / CUDA / cuDNN versions, the determinism
    flags, and the stored/trainable parameter split, which makes a frozen-gate arm
    auditable from its own record.
    """
    dev = torch.device(device)
    gpu: dict[str, Any] = {
        "gpu_name": None,
        "gpu_capability": None,
        "gpu_memory_mb": None,
        "gpu_count": 0,
    }
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index or 0
        props = torch.cuda.get_device_properties(idx)
        gpu = {
            "gpu_name": torch.cuda.get_device_name(idx),
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_memory_mb": round(props.total_memory / (1024 * 1024)),
            "gpu_count": torch.cuda.device_count(),
        }
    return {
        **gpu,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": _cudnn_version(),
        # CUDA determinism is off (utils/seeding.py), so this is recorded rather than assumed
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "n_params_stored": int(sum(p.numel() for p in model.parameters())),
        "n_params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def make_encoder(cfg: Any) -> BatchEncoder:
    """Encoder for this run, named by ``cfg.resolved.encoder``."""
    return BatchEncoder(str(cfg.resolved.encoder), int(cfg.resolved.T))


def train_surrogate(
    cfg: Any,
    model: FeedForwardSNN,
    loaders: Sequence,
    info: Any,
    device: torch.device,
    run_dir: str | Path | None = None,
) -> dict:
    """Train ``model``, writing ``history.csv``; returns the ``metrics.json`` payload."""
    train_loader, test_loader = loaders[0], loaders[1]
    run_dir = Path(run_dir) if run_dir is not None else resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    n_classes = int(info.n_classes)
    loss_name = str(cfg.resolved.loss)
    epochs = int(cfg.resolved.epochs)
    grad_clip = cfg.train.get("grad_clip", None)
    window = cfg.resolved.get("response_window", None)
    window = int(window) if window else None
    limit_train = cfg.train.get("limit_train_batches", None)
    limit_test = cfg.train.get("limit_test_batches", None)
    eval_every = int(cfg.train.get("eval_every", 1) or 1)
    patience, min_delta, tail_len = early_stopping_cfg(cfg)

    deviations: list[str] = [str(d) for d in (cfg.get("deviations", []) or [])]
    encoder = make_encoder(cfg)
    deviations.extend(encoder.deviations)

    optimizer = build_optimizer(cfg, model)
    history = HistoryWriter(run_dir)
    hidden = resolved_hidden(cfg)
    encoder_name = str(cfg.resolved.encoder)
    prov = provenance(device, model)

    print(
        f"[{utcnow()}] run {cfg.protocol.name}/{cfg.dataset.name}/{cfg.model.name}/"
        f"seed{cfg.seed} on {describe_device(device)}"
    )
    print(
        f"  params={model.param_count()}  hidden={hidden}  "
        f"T={int(cfg.resolved.T)}  batch={int(cfg.resolved.batch_size)}  "
        f"lr={float(cfg.resolved.lr):g}  epochs={epochs}  loss={loss_name}  "
        f"encoder={encoder_name}"
    )

    t_start = time.perf_counter()
    last_train = last_test = None
    best_acc = float("-inf")
    used_p_as_mem = False
    history_rows: list[dict] = []
    monitored: list[float] = []      # the MONITOR quantity, one entry per epoch
    stopped_epoch = 0
    criterion_met = False            # the plateau rule fired
    nonfinite_loss = False
    health: dict = {}                # readout failure states of the final scored pass

    for epoch in range(1, epochs + 1):
        tr, sub1 = run_epoch(
            model,
            train_loader,
            encoder=encoder,
            device=device,
            loss_name=loss_name,
            n_classes=n_classes,
            split="train",
            optimizer=optimizer,
            grad_clip=None if grad_clip is None else float(grad_clip),
            limit_batches=None if limit_train is None else int(limit_train),
            window=window,
        )
        last_train = tr
        row = tr.row(epoch, float(cfg.resolved.lr))
        history.append(row)
        history_rows.append(row)
        print("  " + tr.line(epoch))

        monitored.append(float(tr.loss))
        stopped_epoch = epoch
        # NaN loses every comparison, so should_stop can never fire on it and the run
        # would otherwise burn the whole cap and then serialise as null
        nonfinite_loss = not math.isfinite(monitored[-1])
        criterion_met = should_stop(monitored, patience, min_delta)
        stop_now = criterion_met or nonfinite_loss

        # the epoch we stop on is always evaluated, whatever eval_every says
        if epoch % eval_every == 0 or epoch == epochs or stop_now:
            with torch.no_grad(), collect_readout() as scored:
                te, sub2 = run_epoch(
                    model,
                    test_loader,
                    encoder=encoder,
                    device=device,
                    loss_name=loss_name,
                    n_classes=n_classes,
                    split="test",
                    optimizer=None,
                    limit_batches=None if limit_test is None else int(limit_test),
                    window=window,
                )
            last_test = te
            health = scored[-1].readout_health() if scored else {}
            best_acc = max(best_acc, te.acc)
            row = te.row(epoch, float(cfg.resolved.lr))
            history.append(row)
            history_rows.append(row)
            print("  " + te.line(epoch))
            used_p_as_mem = used_p_as_mem or sub2
        used_p_as_mem = used_p_as_mem or sub1

        if nonfinite_loss:
            print(
                f"  stopping at epoch {epoch}/{epochs}: {MONITOR} is {monitored[-1]}; "
                "the run cannot recover and the plateau rule cannot fire on it"
            )
            break
        if criterion_met:
            print(
                f"  early stop at epoch {epoch}/{epochs}: {MONITOR} did not improve by "
                f"more than {min_delta:g} for {patience} epochs"
            )
            break

    train_seconds = time.perf_counter() - t_start

    # ---------------------------------------------------------- convergence evidence
    hit_epoch_cap = stopped_epoch >= epochs
    # the boundary case matters: a rule that first fires on the cap epoch saved nothing
    stopped_early = bool(criterion_met and stopped_epoch < epochs)
    if nonfinite_loss:
        stop_reason = "nonfinite"
    elif not patience:
        stop_reason = "disabled"
    elif stopped_early:
        stop_reason = "criterion"
    else:
        stop_reason = "cap"

    monitor_tail = monitored[-tail_len:]
    tail = tail_stats(monitor_tail)
    finite = [v for v in monitored if math.isfinite(v)]
    monitor_best = min(finite) if finite else None
    monitor_best_epoch = (monitored.index(monitor_best) + 1) if finite else None
    epochs_since_best = (
        stopped_epoch - monitor_best_epoch if monitor_best_epoch is not None else None
    )

    # ------------------------------------------------------------- failure states
    train_loss = None if last_train is None else last_train.loss
    test_loss = None if last_test is None else last_test.loss
    uniform_loss = uniform_output_loss(
        loss_name,
        n_classes,
        window or int(cfg.resolved.T),
    )

    def uniform_gap(value: float | None) -> float | None:
        if uniform_loss is None or value is None or not math.isfinite(float(value)):
            return None
        return float(value) - uniform_loss

    train_uniform_gap = uniform_gap(train_loss)
    test_uniform_gap = uniform_gap(test_loss)
    at_uniform_loss = bool(
        train_uniform_gap is not None
        and abs(train_uniform_gap) <= UNIFORM_LOSS_RTOL * abs(uniform_loss)
    )
    readout_silent = bool(health.get("readout_silent", False))
    single_class = bool(health.get("single_class_predictions", False))
    failure_flags = [
        name
        for name, fired in (
            ("nonfinite_loss", nonfinite_loss),
            ("silent_readout", readout_silent),
            ("single_class_predictions", single_class),
            ("at_uniform_loss", at_uniform_loss),
        )
        if fired
    ]

    if patience:
        deviations.append(
            f"early stopping on {MONITOR} (train.patience={patience}, "
            f"train.min_delta={min_delta:g}); train.epochs={epochs} is a cap, the run "
            f"stopped at epoch {stopped_epoch} (stop_reason={stop_reason}). The rule is a "
            "plateau on a noisy online average, not a convergence test: read tail_slope_t, "
            "epochs_since_best and hit_epoch_cap before calling this converged"
        )
    if hit_epoch_cap and patience:
        deviations.append(
            f"the run reached the train.epochs={epochs} cap, so the time-to-plateau "
            "diagnostic is right-censored. That does NOT make the reported accuracy a "
            "lower bound -- accuracy can fall with further training. Treat the number as "
            "this configuration's result AT THIS BUDGET and nothing more."
        )
    if failure_flags:
        deviations.append("failure states detected: " + ", ".join(failure_flags))
    if used_p_as_mem:
        deviations.append(
            "output layer has no membrane; the Born spiking probability p was used as the "
            "membrane analogue for the CE loss and the membrane readout"
        )

    if bool(cfg.train.get("save_model", False)):
        torch.save(
            {"state_dict": model.state_dict(), "hidden": hidden},
            run_dir / "model.pt",
        )

    layer_cfg = cfg.model.get("layer", {}) or {}
    metrics = {
        "run_id": f"{cfg.protocol.name}/{cfg.dataset.name}/{cfg.model.name}/seed{cfg.seed}",
        "finished_at": utcnow(),
        "protocol": str(cfg.protocol.name),
        "dataset": str(cfg.dataset.name),
        "model": str(cfg.model.name),
        "variant": str(cfg.get("variant", "") or ""),
        "seed": int(cfg.seed),
        "trainer": "surrogate",
        "n_params": model.param_count(),
        "n_params_per_layer": [l.param_count() for l in model.layers],
        "n_params_trainable": prov["n_params_trainable"],
        "hidden": hidden,
        "N": int(layer_cfg.get("N", 1) or 1),
        "N_mem": int(layer_cfg.get("N_mem", 0) or 0),
        "T": int(cfg.resolved.T),
        "batch_size": int(cfg.resolved.batch_size),
        "lr": float(cfg.resolved.lr),
        "epochs": epochs,                 # the cap that was configured
        "stopped_epoch": stopped_epoch,   # the epoch training actually ran to
        "stopped_early": stopped_early,   # the rule fired AND it saved epochs
        "criterion_met": criterion_met,   # the rule fired, whenever that was
        "hit_epoch_cap": hit_epoch_cap,   # True -> the result is right-censored
        "stop_reason": stop_reason,       # disabled | criterion | cap | nonfinite
        "monitor": MONITOR,
        "patience": patience,
        "min_delta": min_delta if patience else None,
        # the tail of the monitored trajectory, so the stopping decision is auditable
        # from metrics.json alone -- every tail_* field below is computed from exactly
        # this list and can be recomputed from it
        "monitor_tail": monitor_tail,
        "monitor_best": monitor_best,
        "monitor_best_epoch": monitor_best_epoch,
        "monitor_last": monitored[-1] if monitored else None,
        "epochs_since_best": epochs_since_best,
        **tail,
        "loss": loss_name,
        "encoder": encoder_name,
        "response_window": window,
        "delay": None if cfg.dataset.get("delay", None) is None else int(cfg.dataset.delay),
        "detach_state": bool(cfg.resolved.detach_state),
        "n_neurons": int(sum(l.out_features for l in model.layers)),
        "n_units": model.total_units(),
        "test_accuracy": None if last_test is None else last_test.acc,
        "test_accuracy_mem": None if last_test is None else last_test.acc_mem,
        "best_test_accuracy": None if best_acc == float("-inf") else best_acc,
        "test_loss": test_loss,
        "train_accuracy": None if last_train is None else last_train.acc,
        "train_loss": train_loss,
        # --- failure states: a silent readout scores exactly chance, and a fit sitting
        # on the uninformed loss has learnt nothing. Both must be visible as such.
        "n_scored": health.get("n_scored"),
        "silent_frac": health.get("silent_frac"),
        "tie_frac": health.get("tie_frac"),
        "n_decided": health.get("n_decided"),
        "accuracy_decided": health.get("accuracy_decided"),
        "silent_frac_mem": health.get("silent_frac_mem"),
        "tie_frac_mem": health.get("tie_frac_mem"),
        "pred_class_fraction": health.get("pred_class_fraction"),
        "readout_silent": readout_silent,
        "single_class_predictions": single_class,
        "uniform_loss": uniform_loss,
        "train_uniform_loss_gap": train_uniform_gap,
        "test_uniform_loss_gap": test_uniform_gap,
        "at_uniform_loss": at_uniform_loss,
        "nonfinite_loss": nonfinite_loss,
        "failure_flags": failure_flags,
        "spikes_per_step": None if last_test is None else last_test.spikes_per_step,
        "per_neuron_rate": None
        if last_test is None
        else per_neuron_rate(last_test.spikes_per_step, model.total_units()),
        "layer_rates": [] if last_test is None else last_test.layer_rates,
        "train_seconds": train_seconds,
        "torch_version": torch.__version__,
        "device": describe_device(device),
        "gpu_name": prov["gpu_name"],
        "provenance": prov,
        "n_train": int(info.n_train),
        "n_test": int(info.n_test),
        "deviations": deviations,
        "history": history_rows,
    }
    print(
        f"  done in {train_seconds:.1f}s | epochs {stopped_epoch}/{epochs}"
        f" ({stop_reason}) | test_accuracy_mem={metrics['test_accuracy_mem']}"
        f" | test_accuracy={metrics['test_accuracy']}"
        f" | spikes/step={metrics['spikes_per_step']}"
        + (f" | warning: {'+'.join(failure_flags)}" if failure_flags else "")
    )
    return metrics
