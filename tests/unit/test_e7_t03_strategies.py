"""E7-T03: strategy rules, proportional costs, the strategy trace and the strategy metrics."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML

from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.evaluation.backtest import run_backtest
from regimekit.evaluation.metrics.registry import resolve
from regimekit.finance import FinanceConfig
from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR
from regimekit.finance.config import StrategyCfg
from regimekit.finance.metrics import hit_rate, max_drawdown, sharpe, turnover
from regimekit.finance.strategies import (
    RegimeLongFlat,
    RegimeTargetVol,
    StrategyHook,
    check_backtest_admissible,
    make_strategy,
    strategy_returns,
    strategy_trace,
)
from regimekit.inference.results import ForecastResult, PosteriorResult
from regimekit.utils.errors import ConfigError
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

DT = 1.0 / TRADING_DAYS_PER_YEAR
ROOT = Path(__file__).resolve().parents[2]
_YAML = YAML(typ="safe")


def _posterior(filtered: np.ndarray, smoothed: np.ndarray | None = None) -> PosteriorResult:
    t = filtered.shape[0]
    return PosteriorResult(
        times=np.arange(t).astype("datetime64[D]").astype("datetime64[ns]"),
        filtered=jnp.asarray(filtered),
        smoothed=jnp.asarray(filtered if smoothed is None else smoothed),
        viterbi=np.zeros(t, dtype=int),
        segments=[],
        augmented=None,  # type: ignore[arg-type]
        loglik=0.0,
        aic=0.0,
        bic=0.0,
        icl=0.0,
    )


def _forecast(var: list[float], mean: list[float]) -> ForecastResult:
    v = np.asarray(var, dtype=np.float64)[None, :]  # (h=1, M)
    return ForecastResult(
        origin_time=np.datetime64("2024-01-02", "ns"),
        horizon_times=np.asarray([np.datetime64("2024-01-03", "ns")]),
        regime_probs=jnp.asarray([[0.5, 0.5]]),
        samples=jnp.zeros((1, 1, v.shape[1])),
        cum_quantiles=jnp.zeros((1, 1, v.shape[1])),
        mean=jnp.asarray(np.asarray(mean, dtype=np.float64)[None, :]),
        var=jnp.asarray(v),
        quantile_levels=(0.5,),
        covariate_forecaster="none",
        seed=0,
    )


# --- AC1: costs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("n_series", [1, 2])
def test_e7_t03_costs_reduce_the_cumulative_return_by_exactly_cost_times_turnover(n_series: int) -> None:
    """AC1: sum(r_free) - sum(r_cost) == cost_bps * 1e-4 * sum|dpos| (<= 1e-12), entering from flat."""
    held = np.array([[1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [-0.5, 1.0], [-0.5, 1.0], [0.25, 0.0]])[:, :n_series]
    y = np.linspace(-0.02, 0.03, held.size).reshape(held.shape)
    free, _ = strategy_returns(held, y, 0.0)
    costly, turnover_ = strategy_returns(held, y, 5.0)
    steps = np.abs(np.diff(np.vstack([np.zeros((1, n_series)), held]), axis=0)).sum()
    assert turnover_.sum() == pytest.approx(steps, abs=1e-15)
    assert free.sum() - costly.sum() == pytest.approx(5.0 * 1e-4 * steps, abs=1e-12)
    np.testing.assert_allclose(free, (held * y).sum(axis=1), rtol=0, atol=1e-15)


def test_e7_t03_trace_columns_equity_and_multi_series_layout() -> None:
    """Trace columns equity and multi series layout."""
    held = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    y = np.array([[0.01, 0.02], [-0.01, 0.01], [0.02, -0.01]])
    times = np.arange(3).astype("datetime64[D]").astype("datetime64[ns]")
    trace = strategy_trace(times, held, y, np.full(3, DT), 10.0)
    assert list(trace.columns) == [
        "fold", "time", "position_0", "position_1", "return", "gross", "equity", "turnover", "dt",
    ]  # fmt: skip
    ret, turn = strategy_returns(held, y, 10.0)
    np.testing.assert_array_equal(trace["return"], ret)
    np.testing.assert_array_equal(trace["turnover"], turn)
    np.testing.assert_allclose(trace["equity"], np.exp(np.cumsum(ret)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(trace["gross"], (held * y).sum(axis=1), rtol=0, atol=1e-15)
    one = strategy_trace(times, held[:, :1], y[:, :1], np.full(3, DT), 0.0)
    assert "position" in one.columns


def test_e7_t03_shape_mismatch_is_a_config_error() -> None:
    """Shape mismatch is a config error."""
    with pytest.raises(ConfigError):
        strategy_returns(np.zeros((3, 2)), np.zeros((3, 1)), 1.0)


# --- AC2: regime_long_flat -----------------------------------------------------------------------


def test_e7_t03_long_flat_matches_the_hand_computed_rule_and_the_strict_inequality() -> None:
    """Flat iff P(regime 0) > 0.6 strictly (a probability of exactly 0.6 stays long), per series."""
    filtered = np.array([[0.9, 0.1], [0.5, 0.5], [0.61, 0.39], [0.6, 0.4]])
    rule = RegimeLongFlat(regime=0, prob_above=0.6, n_series=2)
    np.testing.assert_array_equal(np.asarray(rule.positions(_posterior(filtered), None)),
                                  [[0, 0], [1, 1], [0, 0], [1, 1]])  # fmt: skip
    np.testing.assert_array_equal([np.asarray(rule.decide(jnp.asarray(r), None)) for r in filtered],
                                  [[0, 0], [1, 1], [0, 0], [1, 1]])  # fmt: skip


def test_e7_t03_long_flat_uses_smoothed_probabilities_only_when_asked() -> None:
    """Long flat uses smoothed probabilities only when asked."""
    filtered, smoothed = np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]])
    post = _posterior(filtered, smoothed)
    assert float(RegimeLongFlat(0, 0.6, use="filtered").positions(post, None)[0, 0]) == 0.0
    assert float(RegimeLongFlat(0, 0.6, use="smoothed").positions(post, None)[0, 0]) == 1.0


def test_e7_t03_smoothed_use_warns_in_a_backtest(tmp_path: Path) -> None:
    """The warning is the first thing `run_backtest` does (before any data is prepared or fitted)."""
    smoothed = RegimeLongFlat(0, 0.6, use="smoothed")
    with pytest.warns(UserWarning, match="smoothed"):
        check_backtest_admissible(smoothed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_backtest_admissible(RegimeLongFlat(0, 0.6))
        check_backtest_admissible(RegimeTargetVol(0.1, DT))
    obs = bf.observations(bf.simulate_returns(30))
    cfg = bf.make_config(tmp_path)
    with pytest.warns(UserWarning, match="smoothed"), pytest.raises(ConfigError):  # too short: no fold fits
        run_backtest(cfg, lambda: (obs, []), seed=1, hooks=[StrategyHook(smoothed)])


def test_e7_t03_long_flat_regime_out_of_range_is_a_config_error() -> None:
    """Long flat regime out of range is a config error."""
    with pytest.raises(ConfigError) as err:
        RegimeLongFlat(5, 0.6).decide(jnp.asarray([0.5, 0.5]), None)
    assert err.value.path == "evaluation.strategy.flat_if.regime"


# --- AC3: regime_target_vol ----------------------------------------------------------------------


def test_e7_t03_target_vol_scales_inversely_with_predictive_sigma_and_is_capped() -> None:
    """sigma_pred = sqrt(var / dt) = 0.10, 0.20, 0.40 with target 0.15, cap 1 -> 1 (capped), 0.75, 0.375."""
    rule = RegimeTargetVol(target_vol=0.15, nominal_dt=DT)
    sigmas = np.array([0.10, 0.20, 0.40])
    pos = np.asarray(rule.decide(jnp.zeros(2), _forecast(list(sigmas**2 * DT), [0.001] * 3)))
    np.testing.assert_allclose(pos, [1.0, 0.75, 0.375], rtol=1e-12)
    assert np.all(np.abs(pos) <= 1.0)
    capped = RegimeTargetVol(target_vol=0.15, nominal_dt=DT, cap=0.5)
    np.testing.assert_allclose(
        np.asarray(capped.decide(jnp.zeros(2), _forecast(list(sigmas**2 * DT), [0.001] * 3))),
        [0.5, 0.5, 0.375],
        rtol=1e-12,
    )


def test_e7_t03_target_vol_sign_follows_the_predictive_mean() -> None:
    """Target vol sign follows the predictive mean."""
    var = [0.2**2 * DT] * 2
    long_only = RegimeTargetVol(0.1, DT)
    short_ok = RegimeTargetVol(0.1, DT, allow_short=True)
    fc = _forecast(var, [0.01, -0.01])
    np.testing.assert_allclose(np.asarray(long_only.decide(jnp.zeros(2), fc)), [0.5, 0.0])
    np.testing.assert_allclose(np.asarray(short_ok.decide(jnp.zeros(2), fc)), [0.5, -0.5])
    with pytest.raises(ConfigError, match="forecast"):
        long_only.decide(jnp.zeros(2), None)


def test_e7_t03_target_vol_positions_define_only_the_origin_row() -> None:
    """Target vol positions define only the origin row."""
    post = _posterior(np.full((4, 2), 0.5))
    pos = np.asarray(RegimeTargetVol(0.1, DT).positions(post, _forecast([0.2**2 * DT], [0.01])))
    assert pos.shape == (4, 1)
    np.testing.assert_allclose(pos[:, 0], [0, 0, 0, 0.5])


# --- make_strategy and config --------------------------------------------------------------------


def test_e7_t03_make_strategy_builds_the_configured_rule_and_validates() -> None:
    """Make strategy builds the configured rule and validates."""
    flat = make_strategy(
        StrategyCfg(type="regime_long_flat", flat_if={"regime": 1, "prob_above": 0.7, "use": "smoothed"}),
        DT,
        n_series=2,
    )
    assert flat == RegimeLongFlat(1, 0.7, 2, "smoothed")
    target = make_strategy(
        StrategyCfg(type="regime_target_vol", target_vol=0.12, cap=0.8, allow_short=True), DT
    )
    assert target == RegimeTargetVol(0.12, DT, 0.8, True)
    bad = {
        "evaluation.strategy.flat_if": StrategyCfg(type="regime_long_flat"),
        "evaluation.strategy.flat_if.use": StrategyCfg(
            type="regime_long_flat", flat_if={"regime": 0, "prob_above": 0.5, "use": "oracle"}
        ),
        "evaluation.strategy.flat_if.prob_above": StrategyCfg(
            type="regime_long_flat", flat_if={"regime": 0, "prob_above": 1.5}
        ),
        "evaluation.strategy.target_vol": StrategyCfg(type="regime_target_vol"),
        "evaluation.strategy.cap": StrategyCfg(type="regime_target_vol", target_vol=0.1, cap=1.5),
    }
    for path, cfg in bad.items():
        with pytest.raises(ConfigError) as err:
            make_strategy(cfg, DT)
        assert err.value.path == path


def test_e7_t03_strategy_config_fields_default_validate_and_reach_schema_and_dump(tmp_path: Path) -> None:
    """`cap = 1.0`, `allow_short = False` (CLAUDE.md 3.4); Appendix A stays valid."""
    cfg = StrategyCfg(type="regime_target_vol", target_vol=0.1)
    assert (cfg.cap, cfg.allow_short) == (1.0, False)
    with pytest.raises(ValueError, match="cost_bps"):
        StrategyCfg(type="regime_target_vol", target_vol=0.1, cost_bps=-1.0)
    schema_path = tmp_path / "schema.json"
    export_json_schema(schema_path, FinanceConfig)
    props = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["StrategyCfg"]["properties"]
    assert props["cap"]["default"] == 1.0
    assert props["allow_short"]["default"] is False
    resolved = tmp_path / "config.resolved.yaml"
    dump_resolved(
        load_config(ROOT / "tests" / "fixtures" / "configs" / "appendix_a.yaml", FinanceConfig), resolved
    )
    strategy = _YAML.load(resolved)["evaluation"]["strategy"]
    assert (strategy["cap"], strategy["allow_short"]) == (1.0, False)


# --- AC4: metrics on closed-form fixtures --------------------------------------------------------


def _trace(returns: list[float], positions: list[float] | None = None, gross: list[float] | None = None,
           turnover_: list[float] | None = None, dt: float = DT) -> pd.DataFrame:  # fmt: skip
    n = len(returns)
    return pd.DataFrame(
        {
            "position": [1.0] * n if positions is None else positions,
            "return": returns,
            "gross": returns if gross is None else gross,
            "turnover": [1.0] + [0.0] * (n - 1) if turnover_ is None else turnover_,
            "dt": [dt] * n,
        }
    )


def test_e7_t03_constant_return_gives_zero_drawdown_infinite_sharpe_full_hit_rate() -> None:
    """Constant return gives zero drawdown infinite sharpe full hit rate."""
    trace = _trace([0.001] * 252)
    assert max_drawdown(extras={"strategy": trace}) == 0.0
    assert sharpe(extras={"strategy": trace}) == np.inf
    assert hit_rate(extras={"strategy": trace}) == 1.0
    assert turnover(extras={"strategy": trace}) == pytest.approx(1.0 / (252 * DT))  # one entry per year
    assert sharpe(extras={"strategy": _trace([-0.001] * 5)}) == -np.inf
    assert sharpe(extras={"strategy": _trace([0.0] * 5)}) == 0.0


def test_e7_t03_sharpe_and_drawdown_match_closed_forms() -> None:
    """Sharpe and drawdown match closed forms."""
    r = [0.01, 0.03, 0.02, 0.0]
    expected = np.mean(r) / np.std(r, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe(extras={"strategy": _trace(r)}) == pytest.approx(expected, rel=1e-12)
    # per-step clock (dt = 1): no annualization factor
    assert sharpe(extras={"strategy": _trace(r, dt=1.0)}) == pytest.approx(
        np.mean(r) / np.std(r, ddof=1), rel=1e-12
    )
    # +10 %, -10 %, +10 %, -10 % (log): peak e^0.1, trough 1 -> drawdown 1 - e^-0.1; equity starts at 1
    up_down = _trace([0.1, -0.1, 0.1, -0.1])
    assert max_drawdown(extras={"strategy": up_down}) == pytest.approx(1.0 - np.exp(-0.1), rel=1e-12)
    assert sharpe(extras={"strategy": up_down}) == pytest.approx(0.0, abs=1e-12)
    # a loss from the very start counts against the initial equity of 1
    assert max_drawdown(extras={"strategy": _trace([-0.2, 0.0])}) == pytest.approx(
        1.0 - np.exp(-0.2), rel=1e-12
    )


def test_e7_t03_turnover_and_hit_rate_on_a_hand_checked_fixture() -> None:
    """Positions 1, 0, 1, 1 (gross +, +, -, +): hit rate over the invested rows 2/3; turnover per year."""
    trace = _trace(
        [0.01, 0.0, -0.02, 0.01],
        positions=[1.0, 0.0, 1.0, 1.0],
        gross=[0.01, 0.0, -0.02, 0.01],
        turnover_=[1.0, 1.0, 1.0, 0.0],
    )
    assert hit_rate(extras={"strategy": trace}) == pytest.approx(2.0 / 3.0)
    assert turnover(extras={"strategy": trace}) == pytest.approx(3.0 / (4 * DT))
    assert hit_rate(extras={"strategy": _trace([0.01], positions=[0.0])}) == 0.0


def test_e7_t03_strategy_metrics_resolve_and_refuse_an_empty_trace() -> None:
    """Strategy metrics resolve and refuse an empty trace."""
    names = ["sharpe", "max_drawdown", "turnover", "hit_rate"]
    assert all(m.needs == frozenset({"strategy"}) for m in resolve(names))
    for m in resolve(names):
        with pytest.raises(ConfigError):
            m()
        with pytest.raises(ConfigError):
            m(extras={"strategy": pd.DataFrame()})


# --- AC5: reproducibility ------------------------------------------------------------------------


def test_e7_t03_equity_curve_is_bitwise_reproducible() -> None:
    """Equity curve is bitwise reproducible."""
    rng = np.random.default_rng(4)
    held = rng.uniform(-1, 1, size=(200, 2))
    y = rng.normal(0, 0.01, size=(200, 2))
    times = np.arange(200).astype("datetime64[D]").astype("datetime64[ns]")
    first = strategy_trace(times, held, y, np.full(200, DT), 5.0)
    again = strategy_trace(times, held.copy(), y.copy(), np.full(200, DT), 5.0)
    pd.testing.assert_frame_equal(first, again, check_exact=True)
