"""Config resolution: turn the composed hydra config into the numbers a run needs.

Hydra composes four independent groups (``dataset``, ``model``, ``protocol``,
``train``) that all have an opinion about the same handful of knobs. :func:`resolve_cfg`
applies them in a fixed precedence and writes the answers into ``cfg.resolved``
(and back into the canonical fields, so the saved ``config.yaml`` reads true).

Precedence, most specific first:

``T``
    ``dataset.protocol_overrides.<protocol>.T`` -> ``dataset.T`` -> ``protocol.T``
    (memprobe: ``lead_in + 2 + delay + n_readout``.)
``encoder``
    ``dataset.protocol_overrides.<protocol>.encoder`` -> ``dataset.encoder`` ->
    ``protocol.encoder``.
``hidden``
    ``model.hidden`` -> ``dataset.protocol_overrides.<protocol>.hidden`` ->
    ``protocol.hidden``; null everywhere is an error.
``lr``
    ``train.lr`` -> ``model.lr_by_protocol.<protocol>`` -> ``protocol.lr``
``epochs``
    ``train.epochs`` -> ``train.epochs_by_dataset.<dataset>`` ->
    ``protocol.epochs_by_dataset.<dataset>`` -> ``protocol.epochs``
``batch_size``
    ``train.batch_size`` -> ``protocol.batch_size``
``detach_state``, layer overrides
    ``model.overrides_by_protocol.<protocol>`` -> ``protocol`` -> ``model``

Anything the resolver had to guess, or any place where we knowingly depart from
the SQS paper (Chen, Rajendran and Simeone, arXiv:2506.21324), is appended to
``cfg.deviations`` and ends up in ``metrics.json``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        val = obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)
    except Exception:
        return default
    return default if val is None else val


def _proto_overrides(cfg: DictConfig) -> dict:
    ov = _get(cfg.dataset, "protocol_overrides", None)
    if ov is None:
        return {}
    got = _get(ov, str(cfg.protocol.name), None)
    if got is None:
        return {}
    return dict(OmegaConf.to_container(got, resolve=True))  # type: ignore[arg-type]


def _model_proto_overrides(cfg: DictConfig) -> dict:
    ov = _get(cfg.model, "overrides_by_protocol", None)
    if ov is None:
        return {}
    got = _get(ov, str(cfg.protocol.name), None)
    if got is None:
        return {}
    return dict(OmegaConf.to_container(got, resolve=True))  # type: ignore[arg-type]


def resolved_hidden(cfg: DictConfig) -> int:
    """``cfg.resolved.hidden`` as a plain ``int``."""
    return int(_get(cfg.resolved, "hidden", 0))


def _protocol_yaml_default(protocol: str, key: str, default: Any = None) -> Any:
    """The value ``key`` has in the *packaged* ``conf/protocol/<protocol>.yaml``.

    :func:`resolve_cfg` writes its answers back into ``cfg.protocol``, so this is the
    only way to tell "the protocol's own T" from "a T the user asked for on the
    command line" after the fact. Returns ``default`` if the file is not there.
    """
    path = Path(__file__).parent / "conf" / "protocol" / f"{protocol}.yaml"
    try:
        loaded = OmegaConf.load(path)
    except (OSError, ValueError):  # pragma: no cover - a custom protocol yaml
        return default
    val = loaded.get(key, default) if hasattr(loaded, "get") else default
    return default if val is None else val


def resolve_cfg(cfg: DictConfig) -> DictConfig:
    """Fill in every derived value; returns the same config object."""
    OmegaConf.set_struct(cfg, False)
    deviations: list[str] = list(_get(cfg, "deviations", []) or [])

    proto_name = str(cfg.protocol.name)
    ds_ov = _proto_overrides(cfg)
    model_ov = _model_proto_overrides(cfg)

    # ---------------------------------------------------------------- T
    t = ds_ov.get("T", None)
    if t is None:
        t = _get(cfg.dataset, "T", None)
    if t is None:
        t = _get(cfg.protocol, "T", 5)
    cfg.protocol.T = int(t)

    # ------------------------------------------------------ response window
    # Tasks whose answer may only be read after a cue (memprobe) score the last
    # `response_window` steps ONLY.  Without it the decoder sees the cue frame and a
    # memoryless model can classify it directly, so the task stops measuring memory.
    response_window = ds_ov.get("response_window", _get(cfg.dataset, "response_window", None))
    response_window = int(response_window) if response_window else None

    # ---------------------------------------------------------------- encoder
    encoder = ds_ov.get("encoder", None)
    if encoder is None:
        encoder = _get(cfg.dataset, "encoder", None)
    if encoder is None:
        encoder = _get(cfg.protocol, "encoder", None)
    if encoder is None:
        raise ValueError("no encoder: set dataset.encoder or protocol.encoder")
    if str(encoder) != str(_get(cfg.protocol, "encoder", "")):
        deviations.append(
            f"dataset {cfg.dataset.name}: encoder {str(encoder)!r} overrides the protocol's "
            f"{str(_get(cfg.protocol, 'encoder', ''))!r} (the data already carries a time axis)"
        )
    cfg.protocol.encoder = str(encoder)

    # ---------------------------------------------------------------- hidden
    hidden = _get(cfg.model, "hidden", None)
    if hidden is None:
        hidden = ds_ov.get("hidden", None)
    if hidden is None:
        hidden = _get(cfg.protocol, "hidden", None)
    if hidden is None:
        raise ValueError(
            f"model {cfg.model.name!r}: hidden is null; set model.hidden, protocol.hidden "
            "or dataset.protocol_overrides.<protocol>.hidden"
        )
    hidden = int(hidden)
    cfg.model.hidden = hidden

    # ---------------------------------------------------------------- lr
    lr = _get(cfg.train, "lr", None)
    if lr is None:
        lr = _get(_get(cfg.model, "lr_by_protocol", {}), proto_name, None)
    if lr is None:
        lr = model_ov.get("lr", None)
    if lr is None:
        lr = _get(cfg.protocol, "lr", None)
    if lr is None:
        raise ValueError("no learning rate: set train.lr or protocol.lr")
    cfg.train.lr = float(lr)

    # ---------------------------------------------------------------- epochs
    epochs = _get(cfg.train, "epochs", None)
    if epochs is None:
        epochs = _get(_get(cfg.train, "epochs_by_dataset", {}), str(cfg.dataset.name), None)
    if epochs is None:
        epochs = _get(_get(cfg.protocol, "epochs_by_dataset", {}), str(cfg.dataset.name), None)
    if epochs is None:
        epochs = _get(cfg.protocol, "epochs", None)
    if epochs is None:
        raise ValueError("no epoch count: set train.epochs or protocol.epochs")
    cfg.train.epochs = int(epochs)

    # ---------------------------------------------------------------- batch
    batch = _get(cfg.train, "batch_size", None)
    if batch is None:
        batch = _get(cfg.protocol, "batch_size", 128)
    cfg.train.batch_size = int(batch)

    # ---------------------------------------------------------------- layers
    if "layer" not in cfg.model or cfg.model.layer is None:
        raise ValueError(f"model {cfg.model.name!r} has no `layer` block")
    for key in ("layer", "hidden_layer_overrides", "output_layer_overrides"):
        extra = model_ov.get(key, None)
        if extra:
            base = _get(cfg.model, key, {}) or {}
            merged = OmegaConf.merge(OmegaConf.create(dict(base)), OmegaConf.create(dict(extra)))
            cfg.model[key] = merged
            deviations.append(f"protocol {proto_name}: model.{key} <- {dict(extra)}")

    bias_override = _get(cfg.protocol, "bias_override", None)
    if bias_override is not None:
        cfg.model.layer["bias"] = bool(bias_override)
        deviations.append(
            f"protocol {proto_name}: bias={bool(bias_override)} forced on every layer that "
            "accepts a bias"
        )

    detach = model_ov.get("detach_state", None)
    if detach is None:
        detach = _get(cfg.protocol, "detach_state", None)
    if detach is None:
        detach = _get(cfg.model, "detach_state", False)
    detach = bool(detach)

    # ---------------------------------------------------------------- record
    cfg.resolved = OmegaConf.create(
        {
            "protocol": proto_name,
            "dataset": str(cfg.dataset.name),
            "model": str(cfg.model.name),
            "variant": str(_get(cfg, "variant", "") or ""),
            "T": int(cfg.protocol.T),
            "hidden": cfg.model.hidden,
            "lr": float(cfg.train.lr),
            "epochs": int(cfg.train.epochs),
            "batch_size": int(cfg.train.batch_size),
            "detach_state": detach,
            "encoder": str(encoder),
            "response_window": response_window,
            "loss": str(cfg.protocol.loss),
        }
    )

    # ------------------------------------------------- variant sanity check
    derived = derive_variant(cfg)
    if derived and not str(_get(cfg, "variant", "") or ""):
        message = (
            f"variant is empty but {derived} is non-default; this run shares a directory "
            "with the base run"
        )
        if bool(_get(cfg, "strict_variant", False)):
            raise ValueError(message)
        deviations.append(message)

    cfg.deviations = deviations
    return cfg


def derive_variant(cfg: DictConfig) -> str:
    """The variant tag this config *should* carry, from its non-default axes.

    ``"T12"`` when ``protocol.T`` differs from what this (protocol, dataset) pair
    resolves to on its own, ``"delay8"`` for a memory probe. Joined with ``_``;
    empty for a base run.

    Pure: it reads ``cfg`` (and the packaged protocol yaml) and changes nothing.
    :func:`resolve_cfg` uses it only to *check* that a deviating run was tagged.
    """
    parts: list[str] = []

    t = int(_get(_get(cfg, "resolved", None), "T", _get(cfg.protocol, "T", 5)))
    ds_ov = _proto_overrides(cfg)
    default_t = ds_ov.get("T", None)
    if default_t is None:
        default_t = _get(cfg.dataset, "T", None)
    if default_t is None:
        default_t = _protocol_yaml_default(str(cfg.protocol.name), "T", t)
    if int(default_t) != t:
        parts.append(f"T{t}")

    delay = _get(cfg.dataset, "delay", None)
    if delay is not None:
        parts.append(f"delay{int(delay)}")

    return "_".join(parts)
