"""Shared builders for engine tests (not collected): random model, parameters and dataset."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from hsmmkit.data.contracts import PreparedDataset, PrepStats
from hsmmkit.model.duration.negbin import NegBinDuration
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.spec import ModelSpec
from hsmmkit.model.transition import JumpTransition
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.statespace.index import StateSpace

DT = 1.0 / 252.0


@dataclass
class Scenario:
    """A model, parameters and dataset that go together."""

    spec: ModelSpec
    space: StateSpace
    theta: dict[str, jnp.ndarray]
    data: PreparedDataset


def make_dataset(
    y: np.ndarray,
    dt: np.ndarray,
    mask: np.ndarray | None = None,
    z_pi: np.ndarray | None = None,
    z_d: np.ndarray | None = None,
) -> PreparedDataset:
    """A minimal trading-clock `PreparedDataset` around `y`."""
    t = y.shape[0]
    stats = PrepStats(
        cov_mean=np.zeros(0),
        cov_std=np.ones(0),
        train_window=("2000-01-01", "2001-01-01"),
        cov_names=(),
        transforms={},
        lags={},
    )
    return PreparedDataset(
        times=np.arange(t).astype("datetime64[D]").astype("datetime64[ns]"),
        observations=jnp.asarray(y),
        dt=jnp.asarray(dt),
        mask=jnp.ones(y.shape, dtype=bool) if mask is None else jnp.asarray(mask),
        cov_transition=None if z_pi is None else jnp.asarray(z_pi),
        cov_duration=None if z_d is None else jnp.asarray(z_d),
        cov_emission=None,
        cov_names={"transition": (), "duration": (), "emission": ()},
        clock="trading",
        nominal_dt=float(np.median(dt)),
        prep_stats=stats,
        snapshot_ids=("test",),
    )


def make_scenario(
    n_states: int,
    r: list[int],
    order: int,
    t_len: int,
    *,
    n_cov_pi: int = 0,
    n_cov_d: int = 0,
    initial: str = "uniform",
    seed: int = 0,
    irregular: bool = False,
) -> Scenario:
    """Random GBM + negative-binomial jump model, random data and covariates."""
    rng = np.random.default_rng(seed)
    has_pi = bool(n_cov_pi and n_states >= 3)
    spec = ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GaussianFamily(scale_by_dt=True, ito_correction=True),
        duration=NegBinDuration(r=tuple(r), cov_idx=tuple(range(n_cov_d))),
        transition=JumpTransition(cov_idx=tuple(range(n_cov_pi)) if has_pi else ()),
        order=order,
        duration_conditioning="current",
        engine="expanded",
        d_max=10,
        initial=initial,  # type: ignore[arg-type]
        param_specs=(),
    )
    space = build_expanded(spec)
    r_rows = space.layout.tuples.shape[0]

    theta: dict[str, jnp.ndarray] = {
        "emission.mean": jnp.asarray(rng.normal(0.05, 0.1, size=n_states)),
        "emission.scale": jnp.asarray(np.sort(rng.uniform(0.1, 0.5, size=n_states))),
        "duration.rate": jnp.asarray(rng.uniform(1.0, 20.0, size=n_states)),
    }
    if n_cov_d:
        theta["duration.c"] = jnp.asarray(rng.normal(0.0, 0.3, size=(n_states, n_cov_d)))
    if has_pi:
        theta["transition.a"] = jnp.asarray(rng.normal(size=(r_rows, n_states - 2)))
        theta["transition.b"] = jnp.asarray(rng.normal(0.0, 0.5, size=(r_rows, n_states - 2, n_cov_pi)))
    else:
        theta["transition.pi_rows"] = jnp.asarray(rng.dirichlet(np.ones(n_states - 1), size=r_rows))
    if initial == "free":
        n_phase1 = int(space.phase1_idx.shape[0])
        theta["initial.logits"] = jnp.asarray(rng.dirichlet(np.ones(n_phase1)))

    dt = np.full(t_len, DT)
    if irregular:
        dt = DT * rng.integers(1, 4, size=t_len).astype(float)
    y = rng.normal(0.0, 0.01, size=(t_len, 1))
    z_pi = rng.normal(size=(t_len, n_cov_pi)) if has_pi else None
    z_d = rng.normal(size=(t_len, n_cov_d)) if n_cov_d else None
    return Scenario(spec, space, theta, make_dataset(y, dt, z_pi=z_pi, z_d=z_d))


def make_explicit_scenario(
    n_states: int,
    d_max: int,
    t_len: int,
    *,
    family: str = "negbin",
    n_cov_pi: int = 0,
    n_cov_d: int = 0,
    initial: str = "uniform",
    seed: int = 0,
    irregular: bool = False,
    mean_steps: float | None = None,
) -> Scenario:
    """Random GBM + explicit-duration jump model (`family` in negbin/geometric/gamma/weibull)."""
    from hsmmkit.model.duration.gamma import GammaDuration
    from hsmmkit.model.duration.geometric import GeometricDuration
    from hsmmkit.model.duration.weibull import WeibullDuration
    from hsmmkit.statespace.builders import build_explicit

    rng = np.random.default_rng(seed)
    cov = tuple(range(n_cov_d))
    duration: object
    if family == "negbin":
        r = rng.integers(1, min(3, d_max) + 1, size=n_states)
        duration = NegBinDuration(r=tuple(int(x) for x in r), cov_idx=cov)
    elif family == "geometric":
        duration = GeometricDuration(cov_idx=cov)
    elif family == "gamma":
        duration = GammaDuration(cov_idx=cov)
    elif family == "weibull":
        duration = WeibullDuration(cov_idx=cov)
    else:
        raise ValueError(family)
    has_pi = bool(n_cov_pi and n_states >= 3)
    spec = ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GaussianFamily(scale_by_dt=True, ito_correction=True),
        duration=duration,  # type: ignore[arg-type]
        transition=JumpTransition(cov_idx=tuple(range(n_cov_pi)) if has_pi else ()),
        order=1,
        duration_conditioning="current",
        engine="explicit",
        d_max=d_max,
        initial=initial,  # type: ignore[arg-type]
        param_specs=(),
    )
    space = build_explicit(spec)
    mean = mean_steps if mean_steps is not None else max(2.0, d_max / 3.0)
    theta: dict[str, jnp.ndarray] = {
        "emission.mean": jnp.asarray(rng.normal(0.05, 0.1, size=n_states)),
        "emission.scale": jnp.asarray(np.sort(rng.uniform(0.1, 0.5, size=n_states))),
    }
    if family in ("negbin", "geometric"):
        a = 1.0 / rng.uniform(0.5 * mean, 1.5 * mean, size=n_states)
        theta["duration.rate"] = jnp.asarray(-np.log1p(-np.clip(a, 1e-3, 0.9)) / DT)
    elif family == "gamma":
        theta["duration.shape"] = jnp.asarray(rng.uniform(1.0, 4.0, size=n_states))
        theta["duration.scale"] = jnp.asarray(mean * DT / rng.uniform(1.0, 4.0, size=n_states))
    else:
        theta["duration.shape"] = jnp.asarray(rng.uniform(0.8, 2.5, size=n_states))
        theta["duration.scale"] = jnp.asarray(rng.uniform(0.5 * mean, 1.5 * mean, size=n_states) * DT)
    if n_cov_d:
        theta["duration.c"] = jnp.asarray(rng.normal(0.0, 0.3, size=(n_states, n_cov_d)))
    if has_pi:
        theta["transition.a"] = jnp.asarray(rng.normal(size=(n_states, n_states - 2)))
        theta["transition.b"] = jnp.asarray(rng.normal(0.0, 0.5, size=(n_states, n_states - 2, n_cov_pi)))
    else:
        theta["transition.pi_rows"] = jnp.asarray(rng.dirichlet(np.ones(n_states - 1), size=n_states))
    if initial == "free":
        theta["initial.logits"] = jnp.asarray(rng.dirichlet(np.ones(n_states)))

    dt = np.full(t_len, DT)
    if irregular:
        dt = DT * rng.integers(1, 4, size=t_len).astype(float)
    y = rng.normal(0.0, 0.01, size=(t_len, 1))
    z_pi = rng.normal(size=(t_len, n_cov_pi)) if has_pi else None
    z_d = rng.normal(size=(t_len, n_cov_d)) if n_cov_d else None
    return Scenario(spec, space, theta, make_dataset(y, dt, z_pi=z_pi, z_d=z_d))
