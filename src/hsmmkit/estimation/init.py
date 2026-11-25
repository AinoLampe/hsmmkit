"""Initialization methods for multi-start MAP estimation (E5-T03).

Each method gives a `Params` dict (`theta`, constrained space); `estimation/map.py` converts it to
`u` via `ParamManager.flatten`. All three are clock-neutral: they work in steps via `nominal_dt`, never a
year constant (CLAUDE.md §3.2).
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.transition import build_memory_layout
from hsmmkit.plugins import register
from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params.manager import Params
    from hsmmkit.model.params.spec import ParamSpec
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.utils.typing import Array

_LOCATION_NAMES = frozenset({"emission.mean", "emission.mu", "emission.m", "emission.intercept"})
_SCALE_NAMES = frozenset({"emission.scale", "emission.sigma", "emission.s"})
_COEFFICIENT_NAMES = frozenset({"emission.gamma", "emission.delta", "duration.c", "transition.b"})
_ROLL_WINDOW = 21
_KMEANS_ITERS = 20


def _baseline_theta(spec: ModelSpec) -> Params:
    """`theta0` for `spec`: the config's own `init`/`value` defaults, in constrained space."""
    manager = ParamManager(spec.param_specs)
    return dict(manager.unflatten(manager.init_u()))


def _duration_primary_names(spec: ModelSpec) -> tuple[str, ...]:
    """Every `theta` key `spec.duration.init_from_mean_sojourn` fills in."""
    try:
        probe = spec.duration.init_from_mean_sojourn(jnp.ones(spec.n_states), 1.0)
    except NotImplementedError:
        return ()
    return tuple(probe)


def _seed_from_key(key: Array) -> int:
    """A plain int seed derived deterministically from a JAX key, for the NumPy-only k-means step."""
    return int(jax.random.bits(key, (), dtype=jnp.uint32))


def _rolling_realized_vol(y: np.ndarray, nominal_dt: float, window: int = _ROLL_WINDOW) -> np.ndarray:
    """`(T - window + 1,)`: `sqrt(mean(y**2 over the window) / nominal_dt)`, one value per full window."""
    y2 = y**2
    cumsum = np.concatenate(([0.0], np.cumsum(y2)))
    windowed_mean = (cumsum[window:] - cumsum[:-window]) / window
    return np.sqrt(windowed_mean / nominal_dt)


def _kmeans_1d(
    x: np.ndarray, n_clusters: int, seed: int, n_iter: int = _KMEANS_ITERS
) -> tuple[np.ndarray, np.ndarray]:
    """1-D k-means (k-means++ seeding): `(labels (len(x),) int in [0, n_clusters), centers (n_clusters,))`."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = np.empty(n_clusters)
    centers[0] = x[rng.integers(n)]
    for k in range(1, n_clusters):
        d2 = np.min((x[:, None] - centers[None, :k]) ** 2, axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers[k] = x[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        labels = np.argmin((x[:, None] - centers[None, :]) ** 2, axis=1)
        for k in range(n_clusters):
            members = x[labels == k]
            if members.size:
                centers[k] = members.mean()
    return labels, centers


def _series_values(per_series: np.ndarray, pooled: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """`per_series (N, M)` if the spec has a series axis (`shape == (N, M)`), else `pooled (N,)`."""
    return per_series if len(shape) == 2 else pooled


def _spec_shape(spec: ModelSpec, name: str) -> tuple[int, ...]:
    return next(ps.shape for ps in spec.param_specs if ps.name == name)


def _mean_run_length_steps(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """`(n_clusters,)`: mean run length (in grid steps) of each label, via run-length encoding."""
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [labels.shape[0]]))
    run_labels = labels[starts]
    run_lengths = (ends - starts).astype(np.float64)
    out = np.ones(n_clusters)  # fallback: a cluster with no observed run keeps a 1-step default
    for k in range(n_clusters):
        lens = run_lengths[run_labels == k]
        if lens.size:
            out[k] = lens.mean()
    return out


def _empirical_pi_rows(labels: np.ndarray, spec: ModelSpec) -> np.ndarray | None:
    """`(R, N-1)` empirical jump distribution from run-boundary label transitions, or `None`.

    Only the order-1 marginal (state `i` -> state `j`) is estimated from the k-means label sequence; at
    order > 1 the same per-current-state row is broadcast to every memory tuple sharing that current
    state (the label sequence alone cannot identify higher-order conditional transitions reliably).
    Laplace-smoothed (a one-count prior on every off-diagonal cell) so an unseen transition is never
    exactly zero.
    """
    if "transition.pi_rows" not in {s.name for s in spec.param_specs}:
        return None
    n = spec.n_states
    change = np.flatnonzero(np.diff(labels)) + 1
    run_labels = labels[np.concatenate(([0], change))]
    counts = np.ones((n, n))
    np.fill_diagonal(counts, 0.0)
    for a, b in itertools.pairwise(run_labels):
        if a != b:
            counts[a, b] += 1.0
    trans = counts / counts.sum(axis=1, keepdims=True)

    layout = build_memory_layout(n, spec.order)
    pi_rows = np.empty((layout.tuples.shape[0], n - 1))
    for r, c in enumerate(layout.current_of):
        targets = [j for j in range(n) if j != c]
        pi_rows[r] = trans[c, targets]
    return pi_rows


def observed_rows(data: PreparedDataset) -> PreparedDataset:
    """`data` restricted to the rows where every series is observed (`data` itself if there are none to drop).

    The data-driven init methods (`rolling_kmeans`, `random`) read `observations` directly, so rows with
    `mask=False` -- padding rows of a walk-forward window (E7-T02b) or missing prints -- would enter as
    zeros and bias the starting volatilities. `estimation.map.fit` initializes from this view; the
    objective itself always sees the full, masked dataset.
    """
    keep = np.all(np.asarray(data.mask), axis=-1)
    if bool(keep.all()):
        return data

    def pick(block: Array | None) -> Array | None:
        return None if block is None else jnp.asarray(block)[keep]

    return dataclasses.replace(
        data,
        times=np.asarray(data.times)[keep],
        observations=jnp.asarray(data.observations)[keep],
        dt=jnp.asarray(data.dt)[keep],
        mask=jnp.asarray(data.mask)[keep],
        cov_transition=pick(data.cov_transition),
        cov_duration=pick(data.cov_duration),
        cov_emission=pick(data.cov_emission),
        labels=pick(data.labels),
        boundaries=None,
    )


@register("init", "rolling_kmeans")
def rolling_kmeans(data: PreparedDataset, spec: ModelSpec, key: Array) -> Params:
    """Cluster a rolling scale series (21 steps, `sqrt(mean(y^2) / nominal_dt)`) into `n_states` regimes.

    The clusters (k-means++ with the given key, sorted by scale) give `theta`:
    `sigma`/`mu`-like emission params come from the cluster scale/mean, duration params from each
    cluster's mean run length via `spec.duration.init_from_mean_sojourn`, and (order-1 marginal, see
    `_empirical_pi_rows`) `transition.pi_rows` from run-boundary label transitions. Every other spec
    (coefficients, `nu`, covariate-conditioned transition logits, ...) keeps its config baseline.
    """
    theta = _baseline_theta(spec)
    n = spec.n_states
    y_all = np.asarray(data.observations)
    multi = y_all.shape[1] > 1
    # Several series share one regime process: cluster the pooled (root-mean-square) volatility.
    y = np.sqrt(np.mean(y_all**2, axis=1)) if multi else y_all[:, 0]
    if y.shape[0] <= _ROLL_WINDOW:
        return theta

    rv = _rolling_realized_vol(y, data.nominal_dt)
    labels, centers = _kmeans_1d(rv, n, _seed_from_key(key))
    order = np.argsort(centers)
    relabel = np.empty(n, dtype=np.int64)
    relabel[order] = np.arange(n)
    labels = relabel[labels]
    sigma = np.sort(centers)

    # a mixture's component axis is not a state axis: its location/scale start at the configured baseline
    is_mixture = hasattr(spec.emission, "n_components")
    scale_name = next((s.name for s in spec.param_specs if s.name in _SCALE_NAMES), None)
    if scale_name is not None and not is_mixture:
        if multi:
            rv_m = np.stack(
                [_rolling_realized_vol(y_all[:, m], data.nominal_dt) for m in range(y_all.shape[1])], axis=1
            )  # (W, M)
            sigma_nm = np.array(
                [
                    [
                        rv_m[labels == k, m].mean() if np.any(labels == k) else sigma[k]
                        for m in range(rv_m.shape[1])
                    ]
                    for k in range(n)
                ]
            )
            theta[scale_name] = jnp.asarray(_series_values(sigma_nm, sigma, _spec_shape(spec, scale_name)))
        else:
            theta[scale_name] = jnp.asarray(sigma)

    loc_name = next((s.name for s in spec.param_specs if s.name in _LOCATION_NAMES), None)
    if loc_name is not None and not is_mixture:
        if multi:
            y_win_m = y_all[_ROLL_WINDOW - 1 :] / data.nominal_dt  # (W, M)
            mu_nm = np.array(
                [
                    [
                        y_win_m[labels == k, m].mean() if np.any(labels == k) else 0.0
                        for m in range(y_all.shape[1])
                    ]
                    for k in range(n)
                ]
            )
            theta[loc_name] = jnp.asarray(
                _series_values(mu_nm, mu_nm.mean(axis=1), _spec_shape(spec, loc_name))
            )
        else:
            y_win = y[_ROLL_WINDOW - 1 :] / data.nominal_dt
            mu = np.array([y_win[labels == k].mean() if np.any(labels == k) else 0.0 for k in range(n)])
            theta[loc_name] = jnp.asarray(mu)

    if _duration_primary_names(spec):
        mean_steps = jnp.asarray(_mean_run_length_steps(labels, n))
        theta.update(spec.duration.init_from_mean_sojourn(mean_steps, data.nominal_dt))

    pi_rows = _empirical_pi_rows(labels, spec)
    if pi_rows is not None:
        theta["transition.pi_rows"] = jnp.asarray(pi_rows)

    return theta


def _free_mask(spec: ParamSpec) -> np.ndarray:
    """`(n0,)` bool: which leading-axis (state) entries are in `mode == "free"`."""
    n0 = spec.shape[0] if spec.shape else 1
    modes = spec.mode if isinstance(spec.mode, tuple) else (spec.mode,) * n0
    return np.array([m == "free" for m in modes])


def lattice_jitter(spec: ModelSpec, key: Array) -> Params:
    """Config `init`/lattice values, with `mode == "free"` entries jittered by ~10%.

    Positive-domain entries: multiplicative `exp(0.1 eps)`. `nu_gt_2` (Student-t `nu > 2`): the same
    multiplicative jitter applied to `value - 2` and shifted back (`2 + (value - 2) * exp(0.1 eps)`),
    since scaling `value` itself can jitter it below the `> 2` floor. Real-domain: additive `0.1 eps
    scale` with `scale = |value|` (or `1.0` at `value == 0`). Simplex-domain (`transition.pi_rows`,
    `duration.logits`) and `constraint="increasing"` specs are left at their baseline (jitter could
    break the ordering the `increasing` bijector's inverse requires, or land outside the simplex).
    """
    theta = _baseline_theta(spec)
    keys = iter(jax.random.split(key, max(len(spec.param_specs), 1) * 2))
    for ps in spec.param_specs:
        if ps.constraint == "increasing" or ps.domain in ("simplex_ref", "pmf"):
            continue
        value = theta[ps.name]
        mask = _free_mask(ps)
        if not mask.any():
            continue
        eps = jax.random.normal(next(keys), value.shape)
        if ps.domain == "positive":
            jittered = value * jnp.exp(0.1 * eps)
        elif ps.domain == "nu_gt_2":
            jittered = 2.0 + (value - 2.0) * jnp.exp(0.1 * eps)
        else:
            scale = jnp.where(value == 0, 1.0, jnp.abs(value))
            jittered = value + 0.1 * eps * scale
        mask_b = jnp.asarray(mask).reshape((-1, *([1] * (value.ndim - 1))))
        theta[ps.name] = jnp.where(mask_b, jittered, value)
    return theta


@register("init", "random")
def random(data: PreparedDataset, spec: ModelSpec, key: Array) -> Params:
    """Scale-free, clock-neutral random restarts: `s = sd(observations) / sqrt(nominal_dt)`.

    `sigma ~ U[0.25, 4] s` (sorted ascending), `mu ~ U[-1, 1] s`, mean sojourn `~ LogU[5, 250]` steps
    (per state) converted via `init_from_mean_sojourn`, `transition.pi_rows`/`transition.a` logits
    `~ N(0, 0.5)` (a `simplex_ref` row is `softmax(concat([0], u))`), coefficient blocks `~ N(0, 0.1)`.
    Anything else (e.g. `nu`) keeps its config baseline.
    """
    theta = _baseline_theta(spec)
    n = spec.n_states
    y_all = np.asarray(data.observations)
    multi = y_all.shape[1] > 1
    s_m = np.std(y_all, axis=0) / np.sqrt(data.nominal_dt)  # (M,) per-series scale
    s = float(np.std(y_all[:, 0])) / np.sqrt(data.nominal_dt) if not multi else float(s_m.mean())
    keys = iter(jax.random.split(key, max(len(spec.param_specs), 1) * 2 + 4))

    scale_name = next((sp.name for sp in spec.param_specs if sp.name in _SCALE_NAMES), None)
    if scale_name is not None:
        u01 = jax.random.uniform(next(keys), (n,), minval=0.25, maxval=4.0)
        if multi and len(_spec_shape(spec, scale_name)) == 2:
            theta[scale_name] = jnp.sort(u01)[:, None] * jnp.asarray(s_m)[None, :]
        else:
            theta[scale_name] = jnp.sort(u01 * s)

    loc_name = next((sp.name for sp in spec.param_specs if sp.name in _LOCATION_NAMES), None)
    if loc_name is not None:
        if multi and len(_spec_shape(spec, loc_name)) == 2:
            u = jax.random.uniform(next(keys), (n, y_all.shape[1]), minval=-1.0, maxval=1.0)
            theta[loc_name] = u * jnp.asarray(s_m)[None, :]
        else:
            theta[loc_name] = jax.random.uniform(next(keys), (n,), minval=-1.0, maxval=1.0) * s

    if _duration_primary_names(spec):
        log_m = jax.random.uniform(next(keys), (n,), minval=jnp.log(5.0), maxval=jnp.log(250.0))
        mean_steps = jnp.exp(log_m)
        for name, value in spec.duration.init_from_mean_sojourn(mean_steps, data.nominal_dt).items():
            theta[name] = value

    for ps in spec.param_specs:
        if ps.name == "transition.pi_rows":
            # `ps.shape` is the *full* (R, N-1) probability row (CLAUDE.md §3.1: the reference-dropped
            # bijector has one fewer free coordinate, but `theta`'s own value is the whole row); the
            # logits drawn here must be one column narrower so `concat([ref, logits])` lands back on
            # `ps.shape`, not `ps.shape[-1] + 1` (found while adding E5-T05a's N=3 recovery scenarios:
            # silently undetected for N=2, where the resulting free-coordinate count is 0).
            logits = jax.random.normal(next(keys), (*ps.shape[:-1], ps.shape[-1] - 1)) * 0.5
            ref = jnp.zeros((*logits.shape[:-1], 1))
            theta[ps.name] = jax.nn.softmax(jnp.concatenate([ref, logits], axis=-1), axis=-1)
        elif ps.name == "transition.a":
            theta[ps.name] = jax.random.normal(next(keys), ps.shape) * 0.5
        elif ps.name in _COEFFICIENT_NAMES:
            theta[ps.name] = jax.random.normal(next(keys), ps.shape) * 0.1

    return theta


@register("init", "lattice_jitter")
def _lattice_jitter_init(data: PreparedDataset, spec: ModelSpec, key: Array) -> Params:
    """`lattice_jitter` under the common `(data, spec, key)` signature of registered init methods."""
    return lattice_jitter(spec, key)


@register("init", "labels")
def _labels_init(data: PreparedDataset, spec: ModelSpec, key: Array) -> Params:
    """Initialize from the partial labels (`data.labels`); gated until E12-T06 (`gates.GATES`)."""
    raise ConfigError("the 'labels' init is not available before E12-T06", path="estimation.init")
