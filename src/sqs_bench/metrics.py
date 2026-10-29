"""Decoders, accuracy and spike-activity metrics.

Definitions follow the SQS paper (Chen, Rajendran and Simeone, arXiv:2506.21324):

* **rate decoding** -- predict the output neuron with the largest total spike count
  (summed over time *and* over the neuron's ``N`` spike channels), Sec. V-C.
* **spikes per step** -- "the aggregate spike count across the entire network"
  (hidden + output) averaged over test samples and time steps, Fig. 8(b).
* **per-neuron rate** -- the same quantity divided by the number of spike channels
  in the network (``sum_layers out_features * N``), Fig. 10(b).

The decoders are ``argmax``, and ``argmax`` always returns something. A network whose
output layer has gone silent scores *exactly* chance on a balanced probe, because every
score vector is zero and ``argmax`` then returns class 0. That number is a tie-break
constant, not a decision, and it must not be reportable as "chance".
:meth:`RunMetrics.readout_health` counts how often it happened; :func:`uniform_output_loss`
is the matching check on the loss.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import Tensor

#: A spike readout counts as silent once more than this fraction of scored windows
#: carried no output spikes at all. Half is the line because past it most of the
#: reported accuracy is the tie-break constant rather than anything the network did.
SILENT_FRAC_ALERT = 0.5

#: How close the achieved loss must sit to the exact uninformed loss, relative to that
#: loss, before the fit counts as having learnt nothing. 1e-3 of 3*ln(4) = 4.158883 is
#: 4.2e-3, far below the epoch-to-epoch movement of any run that is still learning.
UNIFORM_LOSS_RTOL = 1e-3


def window_slice(seq: Tensor, window: int | None) -> Tensor:
    """The last ``window`` time steps of a ``[T, ...]`` sequence (all of it if ``None``).

    Tasks with a *response window* -- memprobe, where only the frames after the go
    pulse may be scored -- must not let the decoder or the loss see the cue frame,
    otherwise a memoryless model can classify the cue directly and the task stops
    measuring memory at all.
    """
    if not window:
        return seq
    w = int(window)
    if w > seq.shape[0]:
        raise ValueError(f"response window {w} exceeds T={seq.shape[0]}")
    return seq[-w:]


def rate_scores(spikes_out: Tensor, window: int | None = None) -> Tensor:
    """``[T, B, C, N]`` (or ``[T, B, C]``) spikes -> the ``[B, C]`` score ``argmax`` sees."""
    s = window_slice(spikes_out, window)
    if s.dim() == 4:
        s = s.sum(dim=-1)
    if s.dim() != 3:
        raise ValueError(f"spikes_out must be [T,B,C,N] or [T,B,C], got {tuple(spikes_out.shape)}")
    return s.sum(dim=0)


def rate_decode(spikes_out: Tensor, window: int | None = None) -> Tensor:
    """``[T, B, C, N]`` (or ``[T, B, C]``) spikes -> ``[B]`` predicted class."""
    return rate_scores(spikes_out, window).argmax(dim=-1)


def membrane_scores(mem_seq: Tensor, window: int | None = None) -> Tensor:
    """``[T, B, C]`` membrane -> the ``[B, C]`` sum over time that ``argmax`` sees."""
    mem_seq = window_slice(mem_seq, window)
    if mem_seq.dim() == 4 and mem_seq.shape[-1] == 1:
        mem_seq = mem_seq.squeeze(-1)
    if mem_seq.dim() != 3:
        raise ValueError(f"mem_seq must be [T,B,C], got {tuple(mem_seq.shape)}")
    return mem_seq.sum(dim=0)


def membrane_decode(mem_seq: Tensor, window: int | None = None) -> Tensor:
    """``[T, B, C]`` membrane -> ``[B]`` predicted class (argmax of the sum over time)."""
    return membrane_scores(mem_seq, window).argmax(dim=-1)


def decision_health(scores: Tensor) -> tuple[Tensor, Tensor]:
    """``(silent, tied)`` boolean masks over the batch of a ``[B, C]`` score matrix.

    *silent* -- every class scored exactly zero, so the window carries no evidence at
    all and the prediction is whatever ``argmax`` returns for a flat vector. *tied* --
    the winning score is attained by more than one class, so index order made the
    decision. Both tests are exact comparisons rather than tolerances: spike scores are
    sums of non-negative integers, and a float readout that is *exactly* constant across
    classes is degenerate rather than merely close. Silence implies a tie whenever there
    is more than one class.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be [B, C], got {tuple(scores.shape)}")
    top = scores.max(dim=-1, keepdim=True).values
    tied = (scores == top).sum(dim=-1) > 1
    silent = (scores == 0).all(dim=-1)
    return silent, tied


def uniform_output_loss(
    loss_name: str,
    n_classes: int,
    n_steps: int,
) -> float | None:
    """The exact loss of an output that carries no information about the label.

    A run sitting on this value has learnt nothing at all, however respectable the
    accuracy beside it looks, so the gap to it belongs in the record.

    ``membrane_ce`` sums cross entropy over the scored steps, so equal logits give
    ``n_steps * ln(C)`` exactly -- ``3 * ln 4 = 4.158883`` for the 3-step, 4-class
    memory probe.

    Returns ``None`` when the uninformed value of the objective is not defined here.
    """
    c = int(n_classes)
    if c < 2 or int(n_steps) < 1:
        return None
    name = str(loss_name)
    if name in {"membrane_ce", "mem_ce", "ce_membrane"}:
        return float(n_steps) * math.log(c)
    return None


def accuracy(pred: Tensor, target: Tensor) -> float:
    """Fraction of correct predictions."""
    if pred.numel() == 0:
        return float("nan")
    return float((pred.reshape(-1) == target.reshape(-1).to(pred.device)).float().mean())


def spikes_per_step(n_spikes: Tensor) -> float:
    """``[T, B]`` per-sample network spike counts -> mean over samples and steps."""
    if n_spikes.numel() == 0:
        return float("nan")
    return float(n_spikes.detach().float().mean())


def per_neuron_rate(sps: float, n_units: int) -> float:
    """Spikes per step per spike channel."""
    if not n_units:
        return float("nan")
    return float(sps) / float(n_units)


def layer_rates(n_spikes_layers: list[Tensor], units_per_layer: list[int]) -> list[dict[str, float]]:
    """Per-layer ``spikes_per_step`` and ``per_neuron_rate``."""
    out = []
    for i, n in enumerate(n_spikes_layers):
        units = units_per_layer[i] if i < len(units_per_layer) else 0
        sps = spikes_per_step(n)
        out.append(
            {
                "layer": float(i),
                "units": float(units),
                "spikes_per_step": sps,
                "per_neuron_rate": per_neuron_rate(sps, units),
            }
        )
    return out


def confusion_matrix(pred: Tensor, target: Tensor, n_classes: int) -> Tensor:
    """``[n_classes, n_classes]`` int64 matrix, rows = true, columns = predicted."""
    p = pred.reshape(-1).long().cpu()
    t = target.reshape(-1).long().cpu()
    idx = t * n_classes + p
    return torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


#: Accumulators register here while a :func:`collect_readout` block is open. ``run_epoch``
#: builds its :class:`RunMetrics` internally and returns only a scalar summary, so the
#: readout-health counters have no other way back to the caller; asking for them at the
#: call site costs less than threading a hook through every trainer. Nothing is recorded
#: unless a block is open, and the training loop is single-threaded.
_COLLECTORS: list[list["RunMetrics"]] = []


@contextmanager
def collect_readout() -> Iterator[list["RunMetrics"]]:
    """Capture every :class:`RunMetrics` built inside the block, newest last."""
    bucket: list[RunMetrics] = []
    _COLLECTORS.append(bucket)
    try:
        yield bucket
    finally:
        _COLLECTORS.remove(bucket)


@dataclass
class RunMetrics:
    """Streaming accumulator over the batches of one epoch / evaluation pass.

    All quantities are sample-weighted, so uneven final batches are handled
    correctly.
    """

    n_classes: int
    units_per_layer: list[int] = field(default_factory=list)

    n_samples: int = 0
    n_steps: int = 0  # sum over batches of T * B
    loss_sum: float = 0.0
    correct_spike: int = 0
    correct_mem: int = 0
    have_mem: bool = False
    window: int | None = None       # score only the last `window` steps (response window)
    spike_sum: float = 0.0
    layer_spike_sum: list[float] = field(default_factory=list)
    confusion: Tensor | None = None

    # --- failure-state counters, read back through readout_health()
    n_silent: int = 0          # scored windows whose spike score vector was all zero
    n_tied: int = 0            # decisions index order made, not evidence
    n_decided: int = 0         # the rest
    correct_decided: int = 0
    n_silent_mem: int = 0
    n_tied_mem: int = 0

    def __post_init__(self) -> None:
        for bucket in _COLLECTORS:
            bucket.append(self)

    def update(
        self,
        *,
        targets: Tensor,
        spikes_out: Tensor,
        n_spikes: Tensor | None = None,
        n_spikes_layers: list[Tensor] | None = None,
        mem_out: Tensor | None = None,
        loss: float | Tensor | None = None,
    ) -> None:
        b = int(targets.shape[0])
        self.n_samples += b
        scores = rate_scores(spikes_out, self.window)
        pred = scores.argmax(dim=-1)
        tgt = targets.to(pred.device).reshape(-1)
        correct = pred == tgt
        self.correct_spike += int(correct.sum())
        cm = confusion_matrix(pred, tgt, self.n_classes)
        self.confusion = cm if self.confusion is None else self.confusion + cm

        silent, tied = decision_health(scores)
        decided = ~tied
        self.n_silent += int(silent.sum())
        self.n_tied += int(tied.sum())
        self.n_decided += int(decided.sum())
        self.correct_decided += int((correct & decided).sum())

        if mem_out is not None:
            self.have_mem = True
            mem_scores = membrane_scores(mem_out, self.window)
            self.correct_mem += int((mem_scores.argmax(dim=-1) == tgt).sum())
            silent_mem, tied_mem = decision_health(mem_scores)
            self.n_silent_mem += int(silent_mem.sum())
            self.n_tied_mem += int(tied_mem.sum())

        if loss is not None:
            self.loss_sum += float(loss) * b

        if n_spikes is not None:
            t = int(n_spikes.shape[0])
            self.n_steps += t * b
            self.spike_sum += float(n_spikes.detach().float().sum())
        if n_spikes_layers:
            if not self.layer_spike_sum:
                self.layer_spike_sum = [0.0] * len(n_spikes_layers)
            for i, n in enumerate(n_spikes_layers):
                self.layer_spike_sum[i] += float(n.detach().float().sum())

    # ------------------------------------------------------------------ views
    @property
    def acc(self) -> float:
        return self.correct_spike / self.n_samples if self.n_samples else float("nan")

    @property
    def acc_mem(self) -> float | None:
        if not self.have_mem or not self.n_samples:
            return None
        return self.correct_mem / self.n_samples

    @property
    def loss(self) -> float:
        return self.loss_sum / self.n_samples if self.n_samples else float("nan")

    @property
    def spikes_per_step(self) -> float:
        return self.spike_sum / self.n_steps if self.n_steps else float("nan")

    @property
    def per_neuron_rate(self) -> float:
        return per_neuron_rate(self.spikes_per_step, sum(self.units_per_layer))

    def layer_rates(self) -> list[dict[str, float]]:
        out = []
        for i, s in enumerate(self.layer_spike_sum):
            units = self.units_per_layer[i] if i < len(self.units_per_layer) else 0
            sps = s / self.n_steps if self.n_steps else float("nan")
            out.append(
                {
                    "layer": i,
                    "units": units,
                    "spikes_per_step": sps,
                    "per_neuron_rate": per_neuron_rate(sps, units),
                }
            )
        return out

    def readout_health(self) -> dict:
        """Whether the scored numbers were decisions or tie-breaks.

        ``accuracy_decided`` is the accuracy restricted to windows that carried enough
        evidence for ``argmax`` to have a unique winner, so a silent network cannot hide
        behind the chance-looking figure its tie-breaks produce.
        ``pred_class_fraction`` shows the collapse directly.
        """
        n = self.n_samples
        if not n:
            return {}
        pred_counts = (
            [0.0] * int(self.n_classes)
            if self.confusion is None
            else [float(v) for v in self.confusion.sum(dim=0).tolist()]
        )
        silent_frac = self.n_silent / n
        return {
            "n_scored": n,
            "silent_frac": silent_frac,
            "tie_frac": self.n_tied / n,
            "n_decided": self.n_decided,
            "accuracy_decided": (
                self.correct_decided / self.n_decided if self.n_decided else None
            ),
            "silent_frac_mem": (self.n_silent_mem / n) if self.have_mem else None,
            "tie_frac_mem": (self.n_tied_mem / n) if self.have_mem else None,
            "pred_class_fraction": [c / n for c in pred_counts],
            "readout_silent": bool(silent_frac > SILENT_FRAC_ALERT),
            "single_class_predictions": bool(pred_counts and max(pred_counts) == float(n)),
        }

    def summary(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "loss": self.loss,
            "accuracy": self.acc,
            "accuracy_mem": self.acc_mem,
            "spikes_per_step": self.spikes_per_step,
            "per_neuron_rate": self.per_neuron_rate,
            "layer_rates": self.layer_rates(),
        }
