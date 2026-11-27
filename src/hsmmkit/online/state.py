"""`OnlineState`: the persistable state of a streaming filter (E15-T01).

Concept refs: Part II §14.11, §15.2 (`OnlineState`). The state is the unnormalized forward log-vector
`log_alpha` at step `t` together with the running log-likelihood; saving writes `state.safetensors`
(arrays) and `state.json` (`t`, `last_time`, `spec_hash`, library version). `spec_hash` ties a state to
the fit it came from: `load_state` refuses a state of a different model structure or parameter layout.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

from hsmmkit.utils.errors import PersistenceError

if TYPE_CHECKING:
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.utils.typing import Array

STATE_ARRAYS = "state.safetensors"
STATE_META = "state.json"
STATE_VERSION = 1


@dataclass(frozen=True)
class OnlineState:
    """Streaming filter state (a pytree; `last_time`, `buffer` and `spec_hash` are static metadata).

    Attributes:
        log_alpha: `(S,)` unnormalized forward log-probabilities at step `t` (before the first observation:
            the initial log-distribution).
        log_norm: `()` running log-likelihood `log p(y_1:t)` (`0` before the first observation).
        t: `()` int64, the number of observations processed.
        carry: emission carry; `None` for every family that has no recurrent state.
        last_time: the last timestamp / step index seen (for `dt` under calendar clocks), or `None`.
        buffer: the fixed-lag smoothing ring buffer (E15-T02); `None` here.
        spec_hash: `spec_hash(fit)` of the fit the state belongs to.
    """

    log_alpha: Array
    log_norm: Array
    t: Array
    carry: Any
    last_time: Any
    buffer: Any
    spec_hash: str

    @property
    def step(self) -> int:
        """`t` as a Python int."""
        return int(self.t)


def _flatten(state: OnlineState) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (state.log_alpha, state.log_norm, state.t, state.carry), (
        state.last_time,
        state.buffer,
        state.spec_hash,
    )


def _unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> OnlineState:
    log_alpha, log_norm, t, carry = children
    last_time, buffer, spec_hash_ = aux
    return OnlineState(log_alpha, log_norm, t, carry, last_time, buffer, spec_hash_)


jax.tree_util.register_pytree_node(OnlineState, _flatten, _unflatten)


def spec_hash(fit: FitResult) -> str:
    """sha256 of the model structure (`spec`) and the parameter names and shapes of `fit`."""
    spec: ModelSpec = fit.spec
    payload = {
        "n_states": spec.n_states,
        "n_series": spec.n_series,
        "order": spec.order,
        "d_max": spec.d_max,
        "engine": spec.engine,
        "initial": spec.initial,
        "duration_conditioning": spec.duration_conditioning,
        "memory_mode": spec.memory_mode,
        "n_groups": spec.n_groups,
        "emission": type(spec.emission).__name__,
        "duration": type(spec.duration).__name__,
        "transition": type(spec.transition).__name__,
        "params": [[s.name, list(s.shape)] for s in spec.param_specs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _time_to_json(value: Any) -> Any:  # noqa: ANN401
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return {"kind": "datetime64", "value": str(np.datetime_as_string(value, unit="ns"))}
    if isinstance(value, (int, np.integer)):
        return {"kind": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"kind": "float", "value": float(value)}
    raise PersistenceError(f"cannot save last_time of type {type(value).__name__}", path="state.last_time")


def _time_from_json(blob: Any) -> Any:  # noqa: ANN401
    if blob is None:
        return None
    kind = blob["kind"]
    if kind == "datetime64":
        return np.datetime64(blob["value"], "ns")
    if kind == "int":
        return int(blob["value"])
    if kind == "float":
        return float(blob["value"])
    raise PersistenceError(f"unknown last_time kind {kind!r}", path="state.json")


def save_state(state: OnlineState, path: Path | str) -> Path:
    """Write `state.safetensors` and `state.json` into the directory `path` (created if needed)."""
    if state.buffer is not None:
        raise PersistenceError("saving a fixed-lag buffer is not supported yet", path="state.buffer")
    if state.carry is not None:
        raise PersistenceError("saving an emission carry is not supported yet", path="state.carry")
    from hsmmkit import __version__

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "log_alpha": np.ascontiguousarray(np.asarray(state.log_alpha)),
        "log_norm": np.ascontiguousarray(np.asarray(state.log_norm)),
    }
    save_file(arrays, str(out / STATE_ARRAYS))
    meta = {
        "state_version": STATE_VERSION,
        "t": state.step,
        "last_time": _time_to_json(state.last_time),
        "spec_hash": state.spec_hash,
        "library_version": __version__,
    }
    (out / STATE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_state(path: Path | str, fit: FitResult) -> OnlineState:
    """Read a state saved by `save_state`; it must belong to `fit`.

    Raises:
        PersistenceError: missing or corrupt files, or a `spec_hash` that differs from `fit`'s (the message
            names both hashes).
    """
    src = Path(path)
    try:
        meta = json.loads((src / STATE_META).read_text(encoding="utf-8"))
        arrays = load_file(str(src / STATE_ARRAYS))
    except (OSError, ValueError, KeyError) as exc:
        raise PersistenceError(f"cannot read the online state in {src}: {exc}", path=str(src)) from exc
    expected = spec_hash(fit)
    if meta.get("spec_hash") != expected:
        raise PersistenceError(
            f"the state was saved for another model: spec_hash {meta.get('spec_hash')} "
            f"!= {expected} of the given fit",
            path=str(src / STATE_META),
        )
    return OnlineState(
        log_alpha=jnp.asarray(arrays["log_alpha"]),
        log_norm=jnp.asarray(arrays["log_norm"]),
        t=jnp.asarray(meta["t"], dtype=jnp.int64),
        carry=None,
        last_time=_time_from_json(meta["last_time"]),
        buffer=None,
        spec_hash=expected,
    )
