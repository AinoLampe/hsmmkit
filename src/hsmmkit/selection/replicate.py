"""Replicate datasets simulated from a fitted model (shared by the bootstrap and the PPC; E14-T04/T05)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.inference.simulate import simulate_paths
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec

__all__ = ["simulate_like"]


def _space(spec: ModelSpec) -> Any:  # noqa: ANN401
    return build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)


def simulate_like(theta: Params, spec: ModelSpec, data: PreparedDataset | SequenceBatch, key: Any) -> Any:  # noqa: ANN401
    """`data` with the observations replaced by one simulated path per sequence (same `dt`, covariates)."""
    space = _space(spec)
    seqs = data.sequences if isinstance(data, SequenceBatch) else (data,)
    names = data.group_names if isinstance(data, SequenceBatch) else (None,)
    out = []
    for b, seq in enumerate(seqs):
        th = theta
        if spec.n_groups > 1:
            group = seq.group if seq.group is not None else names[0]
            if group not in names:
                raise ConfigError(f"sequence group {group!r} is not one of {list(names)}", path="group")
            th = {**theta, "initial.logits": theta["initial.logits"][list(names).index(group)]}
        res = simulate_paths(
            jax.random.fold_in(key, b),
            th,
            spec,
            space,
            seq.dt,
            seq.cov_transition,
            seq.cov_duration,
            seq.cov_emission,
            1,
        )
        out.append(
            dataclasses.replace(seq, observations=jnp.asarray(res.values[0]), labels=None, boundaries=None)
        )
    if isinstance(data, SequenceBatch):
        return dataclasses.replace(data, sequences=tuple(out))
    return out[0]
