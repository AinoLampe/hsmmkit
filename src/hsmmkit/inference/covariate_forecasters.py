"""Covariate forecasters (E6-T03): future covariate paths `(n_sim, h, K)` for multi-step forecasts.

Concept refs: §4.9, §5.5. `K` is the union of every covariate column across the three entry points
(transition/duration/emission), de-duplicated by name -- exactly `PreparedDataset.prep_stats.cov_names`
(E2-T07 already builds this de-duplicated tuple). `ColumnMap` maps that unified `(..., K)` array back
to each entry point's own column order (`PreparedDataset.cov_names[entry_point]`), since a name's
position can differ per block and a name may appear in more than one block.

Since covariates are lagged by >= 1 step (CLAUDE.md §3.7), `z_{t+1}` is already known at the forecast
origin `t` and is passed in as `last (K,)`; `h = 1` needs no generation (the caller would not invoke a
forecaster at all, but `sample_paths` handles it gracefully: `last` broadcast with nothing to generate).
`fit(data)` must only look at the training window (`data.prep_stats.train_window`), never the full
series -- the same look-ahead discipline `data.prepare.Preparer` already applies to its own stats.

File I/O (loading a `scenario` path from the `forecast.covariate_forecaster.path` config field) stays
out of this module (`inference/` is not an I/O layer, CLAUDE.md §4 invariant 1); `make_forecaster`
takes already-parsed `scenario_values` instead, leaving the actual file read to the CLI/`Project` layer
that eventually wires this up (E6-T04a/E6-T05).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.plugins import REGISTRY, register
from regimekit.utils.errors import ConfigError, DataError

if TYPE_CHECKING:
    from regimekit.config.schema import CovariateForecasterCfg
    from regimekit.data.contracts import PreparedDataset
    from regimekit.utils.typing import Array

_ENTRY_POINTS = ("transition", "duration", "emission")


class CovariateForecaster(Protocol):
    """Structural type for the four covariate-forecaster families."""

    name: str

    def fit(self, data: PreparedDataset) -> CovariateForecaster:
        """Fit on `data`'s training window only; returns a new, fitted instance."""
        ...

    def sample_paths(self, key: Array, h: int, n_sim: int, last: Array) -> Array:
        """`(n_sim, h, K)`, standardized units; row 0 is `last` (already known at the origin)."""
        ...


@dataclass(frozen=True)
class ColumnMap:
    """Unified `(K,)` covariate columns (`names`) <-> each entry point's own column order."""

    names: tuple[str, ...]
    transition: tuple[int, ...]
    duration: tuple[int, ...]
    emission: tuple[int, ...]

    @classmethod
    def from_dataset(cls, data: PreparedDataset) -> ColumnMap:
        """Build from `data.prep_stats.cov_names` (the union) and `data.cov_names` (per block)."""
        names = data.prep_stats.cov_names
        position = {name: k for k, name in enumerate(names)}
        try:
            blocks = {ep: tuple(position[n] for n in data.cov_names.get(ep, ())) for ep in _ENTRY_POINTS}
        except KeyError as exc:
            raise DataError(
                f"a covariate name in data.cov_names is missing from prep_stats.cov_names: {exc}",
                path="cov_names",
            ) from exc
        return cls(
            names=names,
            transition=blocks["transition"],
            duration=blocks["duration"],
            emission=blocks["emission"],
        )

    def split(self, paths: Array) -> dict[str, Array | None]:
        """`(..., K)` -> `{"transition": (..., K_pi) | None, "duration": ..., "emission": ...}`."""

        def select(idx: tuple[int, ...]) -> Array | None:
            return paths[..., list(idx)] if idx else None

        return {ep: select(getattr(self, ep)) for ep in _ENTRY_POINTS}


def _unified_covariates(data: PreparedDataset) -> np.ndarray:
    """`(T, K)` in `prep_stats.cov_names` order, reading each name from whichever block holds it."""
    names = data.prep_stats.cov_names
    t = int(data.observations.shape[0])
    out = np.full((t, len(names)), np.nan)
    for ep in _ENTRY_POINTS:
        ep_names = data.cov_names.get(ep, ())
        if not ep_names:
            continue
        block = np.asarray(getattr(data, f"cov_{ep}"))
        for local_idx, name in enumerate(ep_names):
            out[:, names.index(name)] = block[:, local_idx]
    if names and np.any(np.isnan(out)):
        raise DataError("a prep_stats.cov_names entry has no covariate block", path="cov_names")
    return out


def _train_window_mask(data: PreparedDataset) -> np.ndarray:
    """Boolean `(T,)`: rows within `prep_stats.train_window` (inclusive on both ends)."""
    start, end = data.prep_stats.train_window
    times = data.times.astype("datetime64[ns]")
    return (times >= np.datetime64(start)) & (times <= np.datetime64(end))


def _seed_from_key(key: Array) -> int:
    """A plain int NumPy seed from a JAX key (the ticket's own NumPy-RNG exception for bootstrap)."""
    return int(jax.random.bits(key, (), dtype=jnp.uint32))


def _bootstrap_source_indices(
    rng: np.random.Generator, t_diff: int, n_steps: int, mean_block_length: float
) -> np.ndarray:
    """`(n_steps,)` indices into a `t_diff`-long series: a stationary (circular) block bootstrap.

    Block starts are uniform over `[0, t_diff)`; block lengths are `Geometric(1 / mean_block_length)`
    (mean `mean_block_length`), truncated at the series' end (circular wrap via `% t_diff`).
    """
    p = 1.0 / mean_block_length
    idx = np.empty(n_steps, dtype=np.int64)
    pos = 0
    while pos < n_steps:
        start = int(rng.integers(0, t_diff))
        block_len = min(int(rng.geometric(p)), n_steps - pos)
        for j in range(block_len):
            idx[pos] = (start + j) % t_diff
            pos += 1
    return idx


@dataclass(frozen=True)
class FreezeForecaster:
    """`freeze`: every row (including row 0) equals `last`."""

    name: str = "freeze"

    def fit(self, data: PreparedDataset) -> FreezeForecaster:
        """No parameters to fit."""
        del data
        return self

    def sample_paths(self, key: Array, h: int, n_sim: int, last: Array) -> Array:
        """`(n_sim, h, K)`, every row `= last`."""
        del key
        return jnp.broadcast_to(last, (n_sim, h, last.shape[-1]))


@dataclass(frozen=True)
class ScenarioForecaster:
    """`scenario`: a user path (by column name) for a subset of columns; missing columns freeze.

    `paths[name]` has length `h` (validated against the `h` passed to `sample_paths`, since `h` is not
    known at construction time) and supplies every row (`0..h-1`) of that column, overriding `last`'s
    own row 0 for the columns it names -- CLAUDE.md §10: the ticket's own AC1 ("row 0 equals last ...
    for all types") is read here as describing the three *stochastic* types (`freeze`/`bootstrap`/`ar`,
    where there is no user override); `scenario`'s own row-0 behaviour is AC4's, not AC1's.
    """

    paths: dict[str, np.ndarray]
    name: str = "scenario"
    column_map: ColumnMap | None = None

    def fit(self, data: PreparedDataset) -> ScenarioForecaster:
        """Resolve column names against `data`'s unified covariate layout."""
        return dataclasses.replace(self, column_map=ColumnMap.from_dataset(data))

    def sample_paths(self, key: Array, h: int, n_sim: int, last: Array) -> Array:
        """`(n_sim, h, K)`: configured columns use the scenario path; others freeze at `last`."""
        del key
        if self.column_map is None:
            raise ConfigError("ScenarioForecaster.fit(data) must be called before sample_paths")
        k = last.shape[-1]
        out = np.asarray(jnp.broadcast_to(last, (n_sim, h, k))).copy()
        for raw_name, values in self.paths.items():
            if raw_name not in self.column_map.names:
                raise ConfigError(
                    f"scenario column {raw_name!r} is not a covariate",
                    path="forecast.covariate_forecaster.path",
                )
            values = np.asarray(values)
            if values.shape[0] != h:
                raise ConfigError(
                    f"scenario column {raw_name!r} has length {values.shape[0]}, expected h={h}",
                    path="forecast.covariate_forecaster.path",
                )
            col = self.column_map.names.index(raw_name)
            out[:, :, col] = values[None, :]
        return jnp.asarray(out)


@dataclass(frozen=True)
class BootstrapForecaster:
    """`bootstrap`: stationary block bootstrap (Politis-Romano) of first differences, from `last`."""

    mean_block_length: float
    name: str = "bootstrap"
    diffs: np.ndarray | None = None  # (T_train - 1, K), training-window first differences

    def fit(self, data: PreparedDataset) -> BootstrapForecaster:
        """First differences of the unified covariates, training window only."""
        z = _unified_covariates(data)[_train_window_mask(data)]
        return dataclasses.replace(self, diffs=np.diff(z, axis=0))

    def sample_paths(self, key: Array, h: int, n_sim: int, last: Array) -> Array:
        """`(n_sim, h, K)`: row 0 is `last`; rows `1..h-1` cumulate bootstrapped differences."""
        if self.diffs is None:
            raise ConfigError("BootstrapForecaster.fit(data) must be called before sample_paths")
        k = last.shape[-1]
        last_np = np.asarray(last)
        out = np.empty((n_sim, h, k))
        out[:, 0, :] = last_np
        if h <= 1:
            return jnp.asarray(out)
        t_diff = self.diffs.shape[0]
        rng = np.random.default_rng(_seed_from_key(key))
        for i in range(n_sim):
            src = _bootstrap_source_indices(rng, t_diff, h - 1, self.mean_block_length)
            out[i, 1:, :] = last_np + np.cumsum(self.diffs[src], axis=0)
        return jnp.asarray(out)


@dataclass(frozen=True)
class ARForecaster:
    """`ar`: VAR(1) `z_{k+1} = c + Phi z_k + eps`, OLS on the training window, Gaussian innovations.

    `diagonal=True` fits each column independently (`Phi` forced diagonal).
    """

    diagonal: bool = False
    name: str = "ar"
    c: np.ndarray | None = None  # (K,)
    phi: np.ndarray | None = None  # (K, K)
    chol_cov: np.ndarray | None = None  # (K, K), Cholesky of the residual covariance

    def fit(self, data: PreparedDataset) -> ARForecaster:
        """OLS VAR(1) (or per-column AR(1) if `diagonal`), training window only."""
        z = _unified_covariates(data)[_train_window_mask(data)]
        k = z.shape[1]
        x = np.column_stack([np.ones(z.shape[0] - 1), z[:-1]])
        y = z[1:]
        if self.diagonal:
            c = np.zeros(k)
            phi = np.zeros((k, k))
            resid = np.empty_like(y)
            for j in range(k):
                xj = np.column_stack([np.ones(z.shape[0] - 1), z[:-1, j]])
                coef, *_ = np.linalg.lstsq(xj, y[:, j], rcond=None)
                c[j], phi[j, j] = coef
                resid[:, j] = y[:, j] - xj @ coef
        else:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            c = coef[0]
            phi = coef[1:].T
            resid = y - x @ coef
        dof = max(resid.shape[0] - x.shape[1], 1)
        cov = (resid.T @ resid) / dof
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(k))
        return dataclasses.replace(self, c=c, phi=phi, chol_cov=chol)

    def sample_paths(self, key: Array, h: int, n_sim: int, last: Array) -> Array:
        """`(n_sim, h, K)`: row 0 is `last`; rows `1..h-1` from the fitted VAR(1) recursion."""
        if self.c is None or self.phi is None or self.chol_cov is None:
            raise ConfigError("ARForecaster.fit(data) must be called before sample_paths")
        k = last.shape[-1]
        n_steps = max(h - 1, 0)
        eps = jax.random.normal(key, (n_sim, n_steps, k))
        eps_corr = jnp.einsum("nhk,jk->nhj", eps, jnp.asarray(self.chol_cov))
        c = jnp.asarray(self.c)
        phi = jnp.asarray(self.phi)

        def step(z_prev: Array, eps_t: Array) -> tuple[Array, Array]:
            z_t = c + phi @ z_prev + eps_t
            return z_t, z_t

        def one_path(eps_path: Array) -> Array:
            _, zs = jax.lax.scan(step, last, eps_path)
            return jnp.concatenate([last[None], zs], axis=0)

        return jax.vmap(one_path)(eps_corr)


@register("covariate_forecaster", "freeze")
def _make_freeze(
    cfg: CovariateForecasterCfg, scenario_values: dict[str, np.ndarray] | None
) -> CovariateForecaster:
    return cast(CovariateForecaster, FreezeForecaster())


@register("covariate_forecaster", "scenario")
def _make_scenario(
    cfg: CovariateForecasterCfg, scenario_values: dict[str, np.ndarray] | None
) -> CovariateForecaster:
    if scenario_values is None:
        raise ConfigError(
            "scenario forecaster needs scenario_values (read from forecast.covariate_forecaster.path)",
            path="forecast.covariate_forecaster.path",
        )
    return cast(CovariateForecaster, ScenarioForecaster(paths=scenario_values))


@register("covariate_forecaster", "bootstrap")
def _make_bootstrap(
    cfg: CovariateForecasterCfg, scenario_values: dict[str, np.ndarray] | None
) -> CovariateForecaster:
    if cfg.mean_block_length is None:
        raise ConfigError(
            "bootstrap forecaster requires mean_block_length",
            path="forecast.covariate_forecaster.mean_block_length",
        )
    return cast(CovariateForecaster, BootstrapForecaster(mean_block_length=float(cfg.mean_block_length)))


@register("covariate_forecaster", "ar")
def _make_ar(
    cfg: CovariateForecasterCfg, scenario_values: dict[str, np.ndarray] | None
) -> CovariateForecaster:
    return cast(CovariateForecaster, ARForecaster(diagonal=bool(cfg.diagonal)))


def make_forecaster(
    cfg: CovariateForecasterCfg, scenario_values: dict[str, np.ndarray] | None = None
) -> CovariateForecaster:
    """`forecast.covariate_forecaster` config -> an (unfitted) `CovariateForecaster`.

    `scenario_values` must already be parsed (the caller reads `cfg.path`, not this function --
    `inference/` is not an I/O layer). The factory is looked up in `REGISTRY` kind `covariate_forecaster`.
    """
    if not REGISTRY.has("covariate_forecaster", cfg.type):
        raise ConfigError(
            f"unknown covariate forecaster type {cfg.type!r}", path="forecast.covariate_forecaster.type"
        )
    factory = REGISTRY.get("covariate_forecaster", cfg.type)
    return cast(CovariateForecaster, factory(cfg, scenario_values))
