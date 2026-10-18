"""E6-T03: covariate forecasters -- `freeze`, `scenario`, `bootstrap`, `ar`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.data.contracts import PreparedDataset, PrepStats
from regimekit.inference.covariate_forecasters import (
    ARForecaster,
    BootstrapForecaster,
    ColumnMap,
    FreezeForecaster,
    ScenarioForecaster,
    _bootstrap_source_indices,
)
from regimekit.utils.errors import ConfigError
from regimekit.utils.rng import key_for


def _make_dataset(z: np.ndarray, names: tuple[str, ...], train_frac: float = 0.8) -> PreparedDataset:
    """A minimal dataset whose unified covariates are `z` (train window = the first `train_frac`)."""
    t = z.shape[0]
    times = np.arange(t).astype("datetime64[D]").astype("datetime64[ns]")
    train_end = str(times[min(int(t * train_frac), t - 1)].astype("datetime64[D]"))
    stats = PrepStats(
        cov_mean=np.zeros(len(names)),
        cov_std=np.ones(len(names)),
        train_window=(str(times[0].astype("datetime64[D]")), train_end),
        cov_names=names,
        transforms={},
        lags={},
    )
    return PreparedDataset(
        times=times,
        observations=jnp.zeros((t, 1)),
        dt=jnp.full((t,), 1.0 / 252),
        mask=jnp.ones((t, 1), dtype=bool),
        cov_transition=jnp.asarray(z),
        cov_duration=jnp.asarray(z[:, :1]),  # shares name[0] with transition: tests column de-dup
        cov_emission=None,
        cov_names={"transition": names, "duration": (names[0],), "emission": ()},
        clock="trading",
        nominal_dt=1.0 / 252,
        prep_stats=stats,
        snapshot_ids=(),
    )


# --- ColumnMap -----------------------------------------------------------------------------------


def test_e6_t03_column_map_dedups_shared_names() -> None:
    """A covariate used by two entry points maps to the same unified column in both."""
    data = _make_dataset(np.zeros((10, 2)), ("x", "y"))
    cmap = ColumnMap.from_dataset(data)
    assert cmap.names == ("x", "y")
    assert cmap.transition == (0, 1)
    assert cmap.duration == (0,)  # "x" shares the transition block's column 0
    assert cmap.emission == ()

    paths = jnp.arange(2 * 3 * 2.0).reshape(2, 3, 2)
    split = cmap.split(paths)
    np.testing.assert_array_equal(np.asarray(split["duration"]), np.asarray(paths[..., [0]]))
    assert split["emission"] is None


# --- AC1: shapes + row 0 == last (freeze/bootstrap/ar) --------------------------------------------


@pytest.mark.parametrize(("h", "n_sim"), [(1, 3), (7, 11)])
def test_e6_t03_shapes_and_row0_equals_last(h: int, n_sim: int) -> None:
    """AC1: `(n_sim, h, K)`; row 0 == `last` for freeze/bootstrap/ar."""
    rng = np.random.default_rng(0)
    t = 400
    z = np.cumsum(rng.normal(0, 0.1, size=(t, 2)), axis=0)
    data = _make_dataset(z, ("x", "y"))
    last = jnp.asarray([0.3, -0.1])

    forecasters = [
        FreezeForecaster().fit(data),
        BootstrapForecaster(mean_block_length=5.0).fit(data),
        ARForecaster().fit(data),
    ]
    for forecaster in forecasters:
        out = forecaster.sample_paths(key_for(0), h, n_sim, last)
        assert out.shape == (n_sim, h, 2)
        np.testing.assert_allclose(
            np.asarray(out[:, 0, :]), np.broadcast_to(np.asarray(last), (n_sim, 2)), atol=1e-10
        )
        assert bool(jnp.all(jnp.isfinite(out)))


def test_e6_t03_freeze_every_row_equals_last() -> None:
    """`freeze`: every row, not just row 0, equals `last`."""
    data = _make_dataset(np.zeros((20, 2)), ("x", "y"))
    last = jnp.asarray([1.0, -2.0])
    out = FreezeForecaster().fit(data).sample_paths(key_for(0), h=5, n_sim=3, last=last)
    for row in range(5):
        np.testing.assert_allclose(np.asarray(out[:, row, :]), np.broadcast_to(np.asarray(last), (3, 2)))


# --- AC2: bootstrap cross-correlation + mean block length ------------------------------------------


def test_e6_t03_bootstrap_mean_block_length_within_10_percent() -> None:
    """AC2: the stationary bootstrap's realized mean block length is within 10% of configured."""
    rng = np.random.default_rng(1)
    mean_block_length = 8.0
    t_diff = 500
    idx = _bootstrap_source_indices(rng, t_diff, n_steps=200_000, mean_block_length=mean_block_length)
    contiguous = (np.diff(idx) % t_diff) == 1
    breaks = np.flatnonzero(~contiguous) + 1
    lengths = np.diff(np.concatenate(([0], breaks, [idx.shape[0]])))
    assert abs(lengths.mean() - mean_block_length) / mean_block_length < 0.10


def test_e6_t03_bootstrap_preserves_cross_correlation_of_differences() -> None:
    """AC2: sample cross-correlation of bootstrapped differences is within 0.1 of the historical one."""
    rng = np.random.default_rng(2)
    t, rho = 3000, 0.6
    eps1 = rng.normal(size=t)
    eps2 = rng.normal(size=t)
    x_diff = eps1
    y_diff = rho * eps1 + np.sqrt(1 - rho**2) * eps2
    z = np.cumsum(np.stack([x_diff, y_diff], axis=1), axis=0)
    data = _make_dataset(z, ("x", "y"), train_frac=1.0)
    historical_corr = float(np.corrcoef(np.diff(z, axis=0).T)[0, 1])

    forecaster = BootstrapForecaster(mean_block_length=10.0).fit(data)
    last = jnp.asarray([0.0, 0.0])
    out = forecaster.sample_paths(key_for(7), h=2, n_sim=5000, last=last)
    sampled_steps = np.asarray(out[:, 1, :])  # one bootstrapped joint difference per path
    sampled_corr = float(np.corrcoef(sampled_steps.T)[0, 1])
    assert abs(sampled_corr - historical_corr) < 0.1


# --- AC3: AR recovers VAR(1) coefficients -----------------------------------------------------------


def test_e6_t03_ar_recovers_var1_coefficients() -> None:
    """AC3: OLS VAR(1) recovers synthetic coefficients within 0.05 at T=5000."""
    rng = np.random.default_rng(3)
    t = 5000
    c_true = np.array([0.01, -0.02])
    phi_true = np.array([[0.5, 0.1], [0.05, 0.4]])
    cov_true = np.array([[0.04, 0.01], [0.01, 0.09]])
    chol_true = np.linalg.cholesky(cov_true)
    z = np.zeros((t, 2))
    for i in range(1, t):
        z[i] = c_true + phi_true @ z[i - 1] + chol_true @ rng.normal(size=2)
    data = _make_dataset(z, ("x", "y"), train_frac=1.0)

    fitted = ARForecaster().fit(data)
    assert fitted.c is not None
    assert fitted.phi is not None
    np.testing.assert_allclose(fitted.c, c_true, atol=0.05)
    np.testing.assert_allclose(fitted.phi, phi_true, atol=0.05)


def test_e6_t03_ar_diagonal_yields_zero_off_diagonals() -> None:
    """AC3: `diagonal=True` forces exact zeros off the diagonal of `phi`."""
    rng = np.random.default_rng(4)
    t = 2000
    z = np.cumsum(rng.normal(0, 0.1, size=(t, 3)), axis=0)
    data = _make_dataset(z, ("x", "y", "z"))
    fitted = ARForecaster(diagonal=True).fit(data)
    assert fitted.phi is not None
    off_diag = fitted.phi - np.diag(np.diagonal(fitted.phi))
    np.testing.assert_array_equal(off_diag, np.zeros((3, 3)))


# --- AC4: scenario ---------------------------------------------------------------------------------


def test_e6_t03_scenario_uses_user_path_and_freezes_missing_columns() -> None:
    """AC4: a configured column follows the user path; a missing column freezes at `last`."""
    data = _make_dataset(np.zeros((30, 2)), ("x", "y"))
    last = jnp.asarray([0.5, -0.5])
    h = 6
    user_path = np.linspace(1.0, 6.0, h)
    forecaster = ScenarioForecaster(paths={"x": user_path}).fit(data)
    out = forecaster.sample_paths(key_for(0), h, n_sim=4, last=last)

    np.testing.assert_allclose(np.asarray(out[:, :, 0]), np.broadcast_to(user_path, (4, h)))
    np.testing.assert_allclose(np.asarray(out[:, :, 1]), np.broadcast_to(float(last[1]), (4, h)))


def test_e6_t03_scenario_wrong_length_raises_config_error() -> None:
    """AC4: a scenario path whose length != h raises `ConfigError`."""
    data = _make_dataset(np.zeros((30, 2)), ("x", "y"))
    forecaster = ScenarioForecaster(paths={"x": np.zeros(3)}).fit(data)
    with pytest.raises(ConfigError):
        forecaster.sample_paths(key_for(0), h=7, n_sim=2, last=jnp.zeros(2))


def test_e6_t03_scenario_unknown_column_raises_config_error() -> None:
    """A scenario path naming a column that isn't a covariate raises `ConfigError`."""
    data = _make_dataset(np.zeros((30, 2)), ("x", "y"))
    forecaster = ScenarioForecaster(paths={"nope": np.zeros(5)}).fit(data)
    with pytest.raises(ConfigError):
        forecaster.sample_paths(key_for(0), h=5, n_sim=2, last=jnp.zeros(2))


# --- AC5: determinism -------------------------------------------------------------------------------


def test_e6_t03_bootstrap_and_ar_are_deterministic_per_key() -> None:
    """AC5: same key -> bitwise identical paths; different key -> differs."""
    rng = np.random.default_rng(5)
    z = np.cumsum(rng.normal(0, 0.1, size=(300, 2)), axis=0)
    data = _make_dataset(z, ("x", "y"))
    last = jnp.asarray([0.1, 0.2])

    for forecaster in (BootstrapForecaster(mean_block_length=4.0).fit(data), ARForecaster().fit(data)):
        out1 = forecaster.sample_paths(key_for(42), h=10, n_sim=5, last=last)
        out2 = forecaster.sample_paths(key_for(42), h=10, n_sim=5, last=last)
        out3 = forecaster.sample_paths(key_for(43), h=10, n_sim=5, last=last)
        np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))
        assert not bool(jnp.allclose(out1, out3))
