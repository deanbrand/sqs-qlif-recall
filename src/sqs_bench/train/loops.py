"""One epoch of training or evaluation.

The encoder is built by the caller and the loss is ``membrane_ce`` on the output
membrane (the Born probability for SQS); both are picked from the config in
:mod:`sqs_bench.train.surrogate`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from sqs_bench.losses import membrane_ce_loss
from sqs_bench.metrics import RunMetrics
from sqs_bench.models.feedforward import FeedForwardSNN, ForwardOut


@dataclass
class EpochResult:
    split: str
    loss: float
    acc: float
    acc_mem: float | None
    spikes_per_step: float
    per_neuron_rate: float
    layer_rates: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    n_samples: int = 0
    n_batches: int = 0

    def row(self, epoch: int, lr: float) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "split": self.split,
            "loss": self.loss,
            "acc": self.acc,
            "acc_mem": "" if self.acc_mem is None else self.acc_mem,
            "spikes_per_step": self.spikes_per_step,
            "per_neuron_rate": self.per_neuron_rate,
            "lr": lr,
            "seconds": self.seconds,
        }

    def line(self, epoch: int) -> str:
        mem = "" if self.acc_mem is None else f" acc_mem {self.acc_mem:.4f}"
        return (
            f"epoch {epoch:>3d} | {self.split:<5s} | loss {self.loss:.4f} | "
            f"acc {self.acc:.4f}{mem} | spikes/step {self.spikes_per_step:.1f} | "
            f"rate {self.per_neuron_rate:.4f} | {self.seconds:.1f}s"
        )


def logits_from(out: ForwardOut) -> tuple[Tensor | None, bool]:
    """``[T, B, C]`` membrane analogue used by the CE loss / membrane readout.

    QLIF neurons expose a membrane. The SQS neuron does not, so the Born spiking
    probability ``p`` (averaged over the neuron's ``N`` channels) stands in for
    it; the second element of the tuple flags that substitution.
    """
    if out.mem_out is not None:
        return out.mem_out, False
    if out.p_out is not None:
        p = out.p_out
        return (p.mean(dim=-1) if p.dim() == 4 else p), True
    return None, False


def compute_loss(
    out: ForwardOut,
    targets: Tensor,
    *,
    loss_name: str,
    n_classes: int,
    window: int | None = None,
) -> tuple[Tensor, Tensor | None, bool]:
    """Returns ``(loss, logits_or_None, used_p_as_membrane)``."""
    logits, substituted = logits_from(out)
    if loss_name in {"membrane_ce", "mem_ce", "ce_membrane"}:
        if logits is None:
            raise RuntimeError(
                "membrane_ce loss needs the output layer to expose 'mem' or 'p' in aux"
            )
        loss = membrane_ce_loss(logits, targets, window=window)
    else:
        raise KeyError(f"unknown loss {loss_name!r}")
    return loss, logits, substituted


def run_epoch(
    model: FeedForwardSNN,
    loader: Iterable,
    *,
    encoder: Callable[[Tensor], Tensor],
    device: torch.device,
    loss_name: str,
    n_classes: int,
    split: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    limit_batches: int | None = None,
    window: int | None = None,
) -> tuple[EpochResult, bool]:
    """Run one pass. ``optimizer`` given -> training pass. Returns ``(result, used_p)``."""
    training = optimizer is not None
    model.train(training)
    acc = RunMetrics(n_classes=n_classes, units_per_layer=model.units_per_layer(), window=window)
    substituted = False
    t0 = time.perf_counter()

    for i, (x, y) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            x_seq = encoder(x)
            out = model(x_seq)
            loss, logits, sub = compute_loss(
                out,
                y,
                loss_name=loss_name,
                n_classes=n_classes,
                window=window,
            )
        substituted = substituted or sub
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        acc.update(
            targets=y,
            spikes_out=out.spikes_out.detach(),
            n_spikes=out.n_spikes,
            n_spikes_layers=out.n_spikes_layers,
            mem_out=None if logits is None else logits.detach(),
            loss=float(loss.detach()),
        )
        del out, loss

    seconds = time.perf_counter() - t0
    res = EpochResult(
        split=split,
        loss=acc.loss,
        acc=acc.acc,
        acc_mem=acc.acc_mem,
        spikes_per_step=acc.spikes_per_step,
        per_neuron_rate=acc.per_neuron_rate,
        layer_rates=acc.layer_rates(),
        seconds=seconds,
        n_samples=acc.n_samples,
    )
    return res, substituted
