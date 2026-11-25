"""E6-T04a: `future_grid` and the Monte-Carlo path samplers (expanded and explicit engines)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR
from regimekit.inference.covariate_forecasters import ColumnMap
from regimekit.inference.forecast import forecast_paths, forecast_paths_explicit
from regimekit.model.spec import ModelSpec
from regimekit.statespace.builders import build_explicit
from regimekit.utils.clock import future_grid
from regimekit.utils.errors import ConfigError
from regimekit.utils.rng import FORECAST_ORIGIN, FORECAST_PATHS, PURPOSE_FORECAST, key_for
from tests.helpers.objective_fixtures import Case, make_case

DT = 1.0 / TRADING_DAYS_PER_YEAR
FRIDAY = np.datetime64("2024-03-29T00:00:00", "ns")


def _weekday(t: np.datetime64) -> int:
    """Monday == 0 ... Sunday == 6."""
    return (int(t.astype("datetime64[D]").astype(np.int64)) + 3) % 7


# --- AC3: future_grid ---------------------------------------------------------------------------


def test_e6_t04a_future_grid_trading_skips_weekends() -> None:
    """AC3: the trading clock skips Saturday/Sunday and gives `1/TRADING_DAYS_PER_YEAR`."""
    times, dt = future_grid(FRIDAY, 4, "trading")
    days = times.astype("datetime64[D]").astype(str).tolist()
    assert days == ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]  # Mon..Thu
    assert np.all(dt == DT)
    assert times.dtype == np.dtype("datetime64[ns]")


def test_e6_t04a_future_grid_calendar_has_calendar_gaps() -> None:
    """AC3: the calendar clock yields the real gaps in days / 365.25."""
    _, dt = future_grid(FRIDAY, 3, "calendar")
    np.testing.assert_allclose(dt * 365.25, [3.0, 1.0, 1.0], atol=1e-12)


def test_e6_t04a_future_grid_steps_is_unit_and_continues_with_step() -> None:
    """AC3: the steps clock yields `1` and continues with the given spacing."""
    step = np.timedelta64(7, "D")
    history = np.array([FRIDAY - step.astype("m8[ns]"), FRIDAY])  # spacing of the observed grid = step
    times, dt = future_grid(FRIDAY, 3, "steps", None, history)
    assert np.all(dt == 1.0)
    assert np.all(np.diff(np.concatenate([[FRIDAY], times])) == step.astype("m8[ns]"))


def test_e6_t04a_future_grid_intraday_bars_skip_the_weekend() -> None:
    """Intraday bars never land on a weekend and use `1/(TRADING_DAYS_PER_YEAR H)`."""
    last = np.datetime64("2024-03-29T18:00:00", "ns")  # Friday evening, 4 bars per day (6 h apart)
    times, dt = future_grid(last, 6, "trading", {"bars_per_day": 4})
    assert all(_weekday(t) < 5 for t in times)
    assert _weekday(times[0]) == 0  # Fri 18:00 + 6 h = Sat 00:00 ... skipped through Sunday to Monday
    assert np.all(dt == 1.0 / (TRADING_DAYS_PER_YEAR * 4))


@pytest.mark.parametrize(("h", "bars"), [(0, 1), (3, 0)])
def test_e6_t04a_future_grid_rejects_bad_arguments(h: int, bars: int) -> None:
    """A zero horizon or zero bars per day is a `ConfigError`."""
    with pytest.raises(ConfigError):
        future_grid(FRIDAY, h, "trading", {"bars_per_day": bars})


# --- helpers ------------------------------------------------------------------------------------


def _gbm_moments(theta: dict[str, jax.Array], dt: float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(theta["emission.mean"]).reshape(-1)
    sigma = np.asarray(theta["emission.scale"]).reshape(-1)
    return (mu - 0.5 * sigma**2) * dt, sigma**2 * dt


def _one_step_regime_probs(
    case: Case, theta: dict[str, jax.Array], log_alpha: jax.Array, z_pi: jax.Array | None = None
) -> np.ndarray:
    z = jnp.zeros((0,)) if z_pi is None else z_pi
    _, log_a = case.space.transition_fn(theta, z, jnp.zeros((0,)), jnp.asarray(DT))
    alpha = np.asarray(jax.nn.softmax(log_alpha))
    return alpha @ np.exp(np.asarray(log_a)) @ np.asarray(case.space.index.collapse)


# --- AC1/AC2: expanded sampler ------------------------------------------------------------------


def test_e6_t04a_h1_regime_distribution_and_mixture_moments() -> None:
    """AC1: h=1 regime frequencies and return moments match the analytic mixture within 3 SE."""
    case = make_case("gbm", "negbin", 1, t_len=30, seed=3)
    theta = case.manager.unflatten(case.u0)
    s = case.space.n_expanded
    log_alpha = jnp.asarray(np.random.default_rng(0).normal(size=s))
    n_sim = 20000
    out = forecast_paths(
        key_for(0, PURPOSE_FORECAST, FORECAST_PATHS),
        theta,
        case.spec,
        case.space,
        log_alpha,
        jnp.asarray([DT]),
        n_sim,
    )
    assert out.states.shape == (n_sim, 1)
    assert out.regimes.shape == (n_sim, 1)
    assert out.returns.shape == (n_sim, 1, 1)

    w = _one_step_regime_probs(case, theta, log_alpha)
    freq = np.bincount(np.asarray(out.regimes[:, 0]), minlength=len(w)) / n_sim
    se = np.sqrt(w * (1 - w) / n_sim)
    assert np.all(np.abs(freq - w) <= 3 * se)

    m, v = _gbm_moments(theta, DT)
    mean = float(w @ m)
    var = float(w @ v + w @ m**2 - (w @ m) ** 2)
    y = np.asarray(out.returns[:, 0, 0])
    assert abs(y.mean() - mean) <= 3 * np.sqrt(var / n_sim)
    se_var = np.sqrt(np.var((y - y.mean()) ** 2) / n_sim)
    assert abs(y.var() - var) <= 3 * se_var


def test_e6_t04a_deterministic_per_key_and_shapes_for_longer_horizon() -> None:
    """AC2: shapes `(n_sim, h)` / `(n_sim, h, M)`; same key bitwise equal, other key differs."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=4)
    theta = case.manager.unflatten(case.u0)
    log_alpha = jnp.zeros(case.space.n_expanded)
    dt = jnp.full(6, DT)
    a = forecast_paths(
        key_for(1, PURPOSE_FORECAST, FORECAST_PATHS), theta, case.spec, case.space, log_alpha, dt, 50
    )
    b = forecast_paths(
        key_for(1, PURPOSE_FORECAST, FORECAST_PATHS), theta, case.spec, case.space, log_alpha, dt, 50
    )
    c = forecast_paths(
        key_for(1, PURPOSE_FORECAST, FORECAST_ORIGIN), theta, case.spec, case.space, log_alpha, dt, 50
    )
    assert a.states.shape == (50, 6)
    assert a.returns.shape == (50, 6, 1)
    assert np.array_equal(np.asarray(a.returns), np.asarray(b.returns))
    assert not np.array_equal(np.asarray(a.returns), np.asarray(c.returns))


def test_e6_t04a_per_path_transition_covariates_drive_the_first_step() -> None:
    """Half the paths see z = +2, half z = -2 at row 0: each half follows its own `A_{T+1}`."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=5, cov_tr=True)
    theta = dict(case.manager.unflatten(case.u0))
    theta["transition.b"] = jnp.full_like(theta["transition.b"], 3.0)
    log_alpha = jnp.log(jnp.asarray([0.2, 0.5, 0.3]))
    n_sim = 20000
    half = n_sim // 2
    z = jnp.concatenate([jnp.full((half, 1, 1), 2.0), jnp.full((half, 1, 1), -2.0)], axis=0)
    colmap = ColumnMap(names=("z",), transition=(0,), duration=(), emission=())
    out = forecast_paths(
        key_for(0, PURPOSE_FORECAST, FORECAST_PATHS),
        theta,
        case.spec,
        case.space,
        log_alpha,
        jnp.asarray([DT]),
        n_sim,
        z,
        colmap,
    )
    reg = np.asarray(out.regimes[:, 0])
    plus = _one_step_regime_probs(case, theta, log_alpha, jnp.asarray([2.0]))
    minus = _one_step_regime_probs(case, theta, log_alpha, jnp.asarray([-2.0]))
    assert np.abs(plus - minus).max() > 0.02  # the covariate matters in this setup
    for sl, p in ((slice(0, half), plus), (slice(half, n_sim), minus)):
        f = np.bincount(reg[sl], minlength=3) / half
        assert np.all(np.abs(f - p) <= 3 * np.sqrt(p * (1 - p) / half) + 1e-12)


def test_e6_t04a_cov_paths_shape_mismatch_raises() -> None:
    """`cov_paths` with the wrong `h` is a `ConfigError`."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=5, cov_tr=True)
    theta = case.manager.unflatten(case.u0)
    colmap = ColumnMap(names=("z",), transition=(0,), duration=(), emission=())
    with pytest.raises(ConfigError):
        forecast_paths(
            key_for(0, PURPOSE_FORECAST),
            theta,
            case.spec,
            case.space,
            jnp.zeros(case.space.n_expanded),
            jnp.full(3, DT),
            10,
            jnp.zeros((10, 2, 1)),
            colmap,
        )


# --- AC4: explicit sampler agrees with the expanded one on a geometric pmf -----------------------


def _explicit_spec(case: Case, d_max: int) -> ModelSpec:
    s = case.spec
    return ModelSpec(
        n_states=s.n_states,
        n_series=1,
        emission=s.emission,
        duration=s.duration,
        transition=s.transition,
        order=1,
        duration_conditioning="current",
        engine="explicit",
        d_max=d_max,
        initial="uniform",
        param_specs=s.param_specs,
    )


def test_e6_t04a_explicit_geometric_matches_expanded_regime_probabilities() -> None:
    """AC4: geometric pmf, explicit and expanded regime probabilities agree within MC error."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=6)
    theta = case.manager.unflatten(case.u0)
    d_max = 300
    spec_x = _explicit_spec(case, d_max)
    space_x = build_explicit(spec_x)
    n = case.spec.n_states
    w0 = np.array([0.7, 0.3])
    elapsed = np.full(d_max, -np.inf)
    elapsed[:20] = -np.log(20)  # elapsed uniform on 1..20 (geometric is memoryless: irrelevant to regimes)
    la_x = jnp.asarray((np.log(w0)[:, None] + elapsed[None, :]).reshape(-1))

    h, n_sim = 10, 20000
    dt = jnp.full(h, DT)
    ex = forecast_paths(
        key_for(0, PURPOSE_FORECAST, FORECAST_PATHS),
        theta,
        case.spec,
        case.space,
        jnp.log(jnp.asarray(w0)),
        dt,
        n_sim,
    )
    im = forecast_paths_explicit(
        key_for(0, PURPOSE_FORECAST, FORECAST_ORIGIN), theta, spec_x, space_x, la_x, dt, n_sim, DT
    )
    assert im.states.shape == (n_sim, h)
    assert im.returns.shape == (n_sim, h, 1)
    assert int(im.states.max()) < n * d_max

    for k in range(h):
        p_e = np.bincount(np.asarray(ex.regimes[:, k]), minlength=n) / n_sim
        p_i = np.bincount(np.asarray(im.regimes[:, k]), minlength=n) / n_sim
        se = np.sqrt((p_e * (1 - p_e) + p_i * (1 - p_i)) / n_sim)
        # 3.5 SE: 10 steps x 2 regimes are compared with fixed keys (multiple-comparison slack over 3 SE)
        assert np.all(np.abs(p_e - p_i) <= 3.5 * se + 1e-12), k


def test_e6_t04a_explicit_origin_elapsed_is_respected() -> None:
    """At `elapsed == d_max` the duration draw is forced to `d_max`: the segment ends after the origin."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=7)
    theta = case.manager.unflatten(case.u0)
    d_max = 5
    spec_x = _explicit_spec(case, d_max)
    space_x = build_explicit(spec_x)
    la = np.full(2 * d_max, -np.inf)
    la[d_max - 1] = 0.0  # regime 0, elapsed == d_max
    out = forecast_paths_explicit(
        key_for(0, PURPOSE_FORECAST), theta, spec_x, space_x, jnp.asarray(la), jnp.full(2, DT), 200, DT
    )
    assert np.all(np.asarray(out.regimes[:, 0]) == 1)  # N=2: leaving regime 0 can only enter regime 1


def _exit_prob(log_pmf_row: np.ndarray, elapsed: int) -> float:
    """`P(d == e | d >= e)` for a 1-based pmf row."""
    p = np.exp(log_pmf_row)
    return float(p[elapsed - 1] / p[elapsed - 1 :].sum())


def test_e6_t04a_explicit_negbin_exit_follows_the_conditional_hazard() -> None:
    """Non-memoryless pmf: the origin duration is conditioned on `d >= elapsed` (hazard, not the pmf)."""
    case = make_case("gbm", "negbin", 1, t_len=30, seed=8)
    theta = case.manager.unflatten(case.u0)
    d_max, elapsed, n_sim = 200, 10, 20000
    spec_x = _explicit_spec(case, d_max)
    space_x = build_explicit(spec_x)
    row = np.asarray(spec_x.duration.log_pmf(theta, d_max, DT, None))[1]  # regime 1 has r == 2 phases
    hazard = _exit_prob(row, elapsed)
    unconditioned = float(np.exp(row[elapsed - 1]))
    se = np.sqrt(hazard * (1 - hazard) / n_sim)
    assert abs(hazard - unconditioned) > 6 * se  # the conditioning is observable in this setup

    la = np.full(2 * d_max, -np.inf)
    la[1 * d_max + elapsed - 1] = 0.0
    out = forecast_paths_explicit(
        key_for(0, PURPOSE_FORECAST, FORECAST_PATHS), theta, spec_x, space_x, jnp.asarray(la),
        jnp.full(1, DT), n_sim, DT,
    )  # fmt: skip
    exit_freq = float(np.mean(np.asarray(out.regimes[:, 0]) == 0))  # N=2: leaving regime 1 enters regime 0
    assert abs(exit_freq - hazard) <= 3 * se


def test_e6_t04a_explicit_origin_segment_uses_its_entry_time_duration_covariate() -> None:
    """The origin segment's pmf is evaluated at the covariate row where that segment began (`T - elapsed`)."""
    case = make_case("gbm", "gamma", 1, t_len=30, seed=9, cov_du=True)
    theta = dict(case.manager.unflatten(case.u0))
    theta["duration.c"] = jnp.full_like(theta["duration.c"], 1.5)
    spec = case.spec
    elapsed, n_sim, t_len = 3, 20000, 30
    z_hist = np.full((t_len, 1), -2.0)
    z_hist[t_len - elapsed, 0] = 2.0  # only the row at the segment start differs
    p_start = _exit_prob(
        np.asarray(spec.duration.log_pmf(theta, spec.d_max, DT, jnp.asarray([2.0])))[0], elapsed
    )
    p_other = _exit_prob(
        np.asarray(spec.duration.log_pmf(theta, spec.d_max, DT, jnp.asarray([-2.0])))[0], elapsed
    )
    se = np.sqrt(p_start * (1 - p_start) / n_sim)
    assert abs(p_start - p_other) > 6 * se

    la = np.full(2 * spec.d_max, -np.inf)
    la[elapsed - 1] = 0.0  # regime 0
    colmap = ColumnMap(names=("z",), transition=(), duration=(0,), emission=())
    out = forecast_paths_explicit(
        key_for(0, PURPOSE_FORECAST, FORECAST_PATHS), theta, spec, case.space, jnp.asarray(la),
        jnp.full(1, DT), n_sim, DT, jnp.zeros((n_sim, 1, 1)), colmap, jnp.asarray(z_hist),
    )  # fmt: skip
    exit_freq = float(np.mean(np.asarray(out.regimes[:, 0]) == 1))
    assert abs(exit_freq - p_start) <= 3 * se


def test_e6_t04a_explicit_duration_covariates_without_history_raise() -> None:
    """Entry-time duration covariates need `cov_paths` and `z_d_hist`; a wrong state count is rejected."""
    case = make_case("gbm", "gamma", 1, t_len=30, seed=9, cov_du=True)
    theta = case.manager.unflatten(case.u0)
    la = jnp.zeros(case.spec.n_states * case.spec.d_max)
    with pytest.raises(ConfigError):
        forecast_paths_explicit(
            key_for(0, PURPOSE_FORECAST), theta, case.spec, case.space, la, jnp.full(1, DT), 5, DT
        )
    with pytest.raises(ConfigError):
        forecast_paths_explicit(
            key_for(0, PURPOSE_FORECAST), theta, case.spec, case.space, la[:-1], jnp.full(1, DT), 5, DT
        )
