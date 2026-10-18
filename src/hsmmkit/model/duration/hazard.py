"""Discrete hazard of an explicit duration pmf (E3-T04)."""

from __future__ import annotations

import jax.numpy as jnp

from regimekit.utils.typing import Array

_TINY = 1e-300


def hazard(log_pmf: Array) -> Array:
    """`h(d) = p(d) / S(d-1)`, `S(d) = 1 - sum_{u<=d} p(u)`, along the last axis (`d = 1..D_max`).

    Computed from the pmf as given (no re-normalization), so an untruncated geometric pmf has the
    constant hazard `a`; for a pmf renormalized over `1..D_max` the last hazard is exactly 1. The
    survival `1 - cumsum` loses relative precision once it falls below ~1e-6 (deep tail);
    it is floored and the result clipped to `[0, 1]`, so the output is finite everywhere.
    """
    p = jnp.exp(log_pmf)
    survival_before = 1.0 - (jnp.cumsum(p, axis=-1) - p)  # S(d-1)
    h = p / jnp.maximum(survival_before, _TINY)
    return jnp.clip(h, 0.0, 1.0)
