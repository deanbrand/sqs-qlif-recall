"""Build a :class:`~sqs_bench.models.feedforward.FeedForwardSNN` from a hydra config.

A model config looks like::

    name: sqsnn
    hidden: null                 # null -> resolved by config.resolve_cfg
    layer:                       # shared by both layers
      _target_: sqs_bench.neurons.sqs.SQSLayer
      N: 1
      N_mem: 1
    hidden_layer_overrides: {}   # merged on top of `layer` for the hidden layer
    output_layer_overrides: {}   # merged on top of `layer` for the output layer

``hidden = H > 0`` builds ``D -> H -> C``; ``hidden = 0`` (or ``None``) builds a
single-layer network ``D -> C``.

Layer classes are located from the ``_target_`` string exactly as hydra would
(``pkg.module.Class``), then called positionally as
``Class(in_features, out_features, **kwargs)`` -- the constructor signature every
:class:`~sqs_bench.neurons.base.SpikingLayer` shares.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

from omegaconf import DictConfig, OmegaConf

from sqs_bench.models.feedforward import FeedForwardSNN


def locate(path: str) -> Any:
    """Import ``pkg.module.Attr`` (hydra ``_target_`` syntax)."""
    if not path or "." not in path:
        raise ValueError(f"_target_ must be a dotted path, got {path!r}")
    module_name, _, attr = path.rpartition(".")
    while True:
        try:
            module = importlib.import_module(module_name)
            break
        except ImportError:
            module_name, _, head = module_name.rpartition(".")
            if not module_name:
                raise
            attr = f"{head}.{attr}"
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, (DictConfig,)):
        return dict(OmegaConf.to_container(obj, resolve=True))  # type: ignore[arg-type]
    return dict(obj)


def _drop_unsupported(cls: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Silently drop kwargs the layer does not accept (e.g. ``bias`` for SQS).

    Returns the filtered kwargs and the names dropped, so the caller can log a
    deviation instead of crashing a whole sweep on one incompatible override.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return kwargs, []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs, []
    accepted = set(sig.parameters) - {"self"}
    kept = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(kept))
    return kept, dropped


def build_layer(
    layer_cfg: Any,
    in_features: int,
    out_features: int,
    overrides: Any = None,
    *,
    dropped: list[str] | None = None,
) -> Any:
    kwargs = _as_dict(layer_cfg)
    kwargs.update(_as_dict(overrides))
    target = kwargs.pop("_target_", None)
    if target is None:
        raise ValueError("layer config needs a _target_ class path")
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    cls = locate(str(target))
    kwargs, drop = _drop_unsupported(cls, kwargs)
    if drop and dropped is not None:
        dropped.append(f"{target}: ignored kwarg(s) {drop}")
    return cls(in_features, out_features, **kwargs)


def hidden_widths(hidden: int | None) -> list[int]:
    """``None`` / ``0`` -> ``[]`` (a single ``D -> C`` layer); ``H`` -> ``[H]``."""
    if hidden is None:
        return []
    return [int(hidden)] if int(hidden) else []


def build_model(
    model_cfg: Any,
    *,
    in_features: int,
    n_classes: int,
    hidden: int | None,
    detach_state: bool = False,
    record_hidden: bool = True,
    record_aux: bool = False,
    dropped: list[str] | None = None,
) -> FeedForwardSNN:
    """Instantiate the spiking network ``[in_features, hidden, n_classes]``.

    The hidden layer gets ``hidden_layer_overrides``, the last one gets
    ``output_layer_overrides``. ``hidden=None`` or ``0`` gives a single layer.
    """
    layer_cfg = model_cfg["layer"] if "layer" in model_cfg else None
    if layer_cfg is None:
        raise ValueError("model config needs a `layer` block with a _target_")
    hid_over = model_cfg.get("hidden_layer_overrides", None)
    out_over = model_cfg.get("output_layer_overrides", None)

    sizes = [int(in_features), *hidden_widths(hidden), int(n_classes)]
    layers = []
    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        layers.append(
            build_layer(
                layer_cfg,
                sizes[i],
                sizes[i + 1],
                out_over if last else hid_over,
                dropped=dropped,
            )
        )

    # a hidden layer with N_out channels feeds an output layer expecting N_in;
    # only 1 -> N broadcasting is allowed, anything else is a config error.
    for a, b in zip(layers, layers[1:]):
        n_out = int(getattr(a, "n_out_channels", 1))
        n_in = int(getattr(b, "n_in_channels", 1))
        if n_out != n_in and n_out != 1:
            raise ValueError(
                f"channel mismatch: layer emits {n_out} channels, next layer expects {n_in}"
            )
    return FeedForwardSNN(
        layers,
        detach_state=detach_state,
        record_hidden=record_hidden,
        record_aux=record_aux,
    )
