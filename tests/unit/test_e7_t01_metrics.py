"""E7-T01: metric registry, predictive/model/label metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from regimekit.config.schema import CovariateForecasterCfg, ForecastSection
from regimekit.config.yaml_io import load_config
from regimekit.evaluation.metrics.labels import balanced_accuracy_matched
from regimekit.evaluation.metrics.predictive import (
    crps_from_samples,
    kde_log_density,
    one_step_mixture_weights,
)
from regimekit.evaluation.metrics.registry import FunctionMetric, available, register, resolve
from regimekit.finance import FinanceConfig
from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR
from regimekit.inference.decode import decode
from regimekit.inference.forecast import MonteCarloForecaster
from regimekit.inference.results import ForecastResult
from regimekit.utils.errors import ConfigError
from tests.helpers.forecast_fixtures import fit_of
from tests.helpers.objective_fixtures import make_case

APPENDIX_A = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "appendix_a.yaml"
DT = 1.0 / TRADING_DAYS_PER_YEAR


def forecast_of_samples(samples: np.ndarray, levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> ForecastResult:
    """A `ForecastResult` around `samples (n, h, M)` with consistent quantiles and moments."""
    h = samples.shape[1]
    cum = np.cumsum(samples, axis=1)
    return ForecastResult(
        origin_time=np.datetime64("2024-01-05", "ns"),
        horizon_times=(np.datetime64("2024-01-08", "ns") + np.arange(h) * np.timedelta64(1, "D")).astype(
            "datetime64[ns]"
        ),
        regime_probs=jnp.full((h, 2), 0.5),
        samples=jnp.asarray(samples),
        cum_quantiles=jnp.asarray(np.transpose(np.quantile(cum, levels, axis=0), (1, 0, 2))),
        mean=jnp.asarray(cum.mean(axis=0)),
        var=jnp.asarray(cum.var(axis=0)),
        quantile_levels=levels,
        covariate_forecaster="none",
        seed=0,
    )


def metric(name: str) -> FunctionMetric:
    """The registered metric `name`."""
    return resolve([name])[0]  # type: ignore[return-value]


# --- AC1: CRPS -----------------------------------------------------------------------------------


def test_e7_t01_crps_matches_the_gaussian_closed_form_within_one_percent() -> None:
    """AC1: sample CRPS of N(mu, sigma^2) draws vs `sigma [z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`."""
    rng = np.random.default_rng(0)
    mu, sigma, y = 0.3, 0.8, 1.1
    x = rng.normal(mu, sigma, size=20000)
    z = (y - mu) / sigma
    exact = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert abs(crps_from_samples(x, y) - exact) <= 0.01 * exact

    fc = forecast_of_samples(x[None, :, None].transpose(1, 0, 2).reshape(20000, 1, 1))
    got = metric("crps")(forecast=fc, realized=np.asarray([[y]]), horizon=1)
    assert abs(got - exact) <= 0.01 * exact  # type: ignore[operator]


def test_e7_t01_sorted_sample_crps_equals_the_quadratic_formula() -> None:
    """The O(n log n) form is the exact V-statistic: it equals the O(n^2) double sum."""
    x = np.random.default_rng(1).normal(size=60)
    y = 0.4
    brute = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert crps_from_samples(x, y) == pytest.approx(brute, abs=1e-12)


def test_e7_t01_kde_log_score_is_close_to_the_true_log_density() -> None:
    """Silverman-bandwidth KDE log density of N(0,1) samples at y=0.5 vs the exact one."""
    x = np.random.default_rng(2).normal(size=20000)
    assert abs(kde_log_density(x, 0.5) - stats.norm.logpdf(0.5)) < 0.03
    fc = forecast_of_samples(x.reshape(-1, 1, 1))
    assert metric("log_score")(forecast=fc, realized=np.asarray([[0.5]]), horizon=1) == pytest.approx(
        kde_log_density(x, 0.5)
    )


# --- AC2: exact mixture --------------------------------------------------------------------------


def test_e7_t01_log_score_mixture_equals_the_analytic_mixture_density() -> None:
    """AC2: `log sum_i w_i N(y; (mu_i - sigma_i^2/2) dt, sigma_i sqrt(dt))` to 1e-8."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=3)
    fit = fit_of(case)
    w = np.asarray([0.35, 0.65])
    base = forecast_of_samples(np.zeros((4, 1, 1)))
    fc = ForecastResult(**{**base.__dict__, "regime_probs": jnp.asarray(w[None, :])})
    y = 0.004
    mu = np.asarray(fit.params["emission.mean"]).reshape(-1)
    sigma = np.asarray(fit.params["emission.scale"]).reshape(-1)
    density = np.sum(w * stats.norm.pdf(y, (mu - 0.5 * sigma**2) * DT, sigma * np.sqrt(DT)))
    got = metric("log_score_mixture")(forecast=fc, fit=fit, realized=np.asarray([[y]]), dt_next=DT, horizon=1)
    assert got == pytest.approx(float(np.log(density)), abs=1e-8)
    with pytest.raises(ConfigError):
        metric("log_score_mixture")(forecast=fc, fit=fit, realized=np.asarray([[y]]), horizon=1)


# --- AC3: PIT ------------------------------------------------------------------------------------


def _pit_and_coverage(scale_forecast: float, n_folds: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    pits, hits = [], []
    for _ in range(n_folds):
        fc = forecast_of_samples(rng.normal(0.0, scale_forecast, size=(2000, 1, 1)))
        out = metric("pit_coverage")(forecast=fc, realized=np.asarray([[rng.normal()]]), horizon=1)
        pits.append(out["pit"])  # type: ignore[call-overload]
        hits.append(out["coverage_90"])  # type: ignore[call-overload]
    return np.asarray(pits), np.asarray(hits)


def test_e7_t01_pit_of_the_true_distribution_is_uniform_and_coverage_is_nominal() -> None:
    """AC3: true-model samples -> KS p > 0.05 over 500 folds; the 90 % interval hit rate is ~0.9."""
    pits, hits = _pit_and_coverage(1.0)
    assert stats.kstest(pits, "uniform").pvalue > 0.05
    se = np.sqrt(0.9 * 0.1 / pits.shape[0])
    assert abs(hits.mean() - 0.9) <= 3 * se


def test_e7_t01_pit_detects_a_wrong_forecast_scale() -> None:
    """Negative control: a forecast 3x too wide is rejected by the same KS test and over-covers."""
    pits, hits = _pit_and_coverage(3.0)
    assert stats.kstest(pits, "uniform").pvalue < 0.01
    assert hits.mean() > 0.97


# --- AC4: registry -------------------------------------------------------------------------------


def test_e7_t01_unknown_metric_name_is_a_config_error_with_path() -> None:
    """AC4: `resolve(["log_score", "nope"])` raises `ConfigError(path="evaluation.metrics")`."""
    with pytest.raises(ConfigError) as err:
        resolve(["log_score", "nope"])
    assert err.value.path == "evaluation.metrics"
    assert "nope" in str(err.value)


def test_e7_t01_every_appendix_a_metric_resolves() -> None:
    """AC4: all names in Appendix A's `evaluation.metrics` resolve, in order."""
    names = load_config(APPENDIX_A, FinanceConfig).evaluation.metrics
    assert [m.name for m in resolve(list(names))] == list(names)
    for required in ("log_score", "log_score_mixture", "crps", "pit_coverage", "rmse", "mae", "aic", "bic"):
        assert required in available()
    assert "balanced_accuracy" in available()


def test_e7_t01_register_rejects_duplicates_unless_replacing() -> None:
    """A second registration of a name raises; `replace=True` swaps the implementation."""
    original = metric("mae")
    with pytest.raises(ConfigError):
        register(FunctionMetric("mae", frozenset({"forecast"}), lambda **_: 0.0))
    try:
        register(FunctionMetric("mae", frozenset({"forecast"}), lambda **_: 123.0), replace=True)
        assert metric("mae")() == 123.0
    finally:
        register(original, replace=True)


def test_e7_t01_strategy_metrics_declare_their_need_and_refuse_to_run() -> None:
    """`sharpe`/`max_drawdown` resolve (Appendix A), declare `strategy` and refuse to run without a trace."""
    for name in ("sharpe", "max_drawdown"):
        m = metric(name)
        assert m.needs == frozenset({"strategy"})
        with pytest.raises(ConfigError):
            m()


# --- AC5: balanced accuracy ----------------------------------------------------------------------


def test_e7_t01_balanced_accuracy_is_invariant_to_label_permutation() -> None:
    """AC5: relabelling either the regimes or the labels does not change the score."""
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 3, size=300)
    predicted = np.where(rng.random(300) < 0.8, labels, rng.integers(0, 3, size=300))
    base = balanced_accuracy_matched(predicted, labels)
    for perm in ([1, 2, 0], [2, 0, 1], [0, 2, 1]):
        assert balanced_accuracy_matched(np.asarray(perm)[predicted], labels) == pytest.approx(base)
        assert balanced_accuracy_matched(predicted, np.asarray(perm)[labels]) == pytest.approx(base)
    assert base > 0.7


def test_e7_t01_balanced_accuracy_hand_checked_values_and_limits() -> None:
    """Perfect up to relabelling = 1; a hand-worked imbalanced case; >6 classes / bad lengths raise."""
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    assert balanced_accuracy_matched(1 - labels, labels) == 1.0
    # predicted: 3 of 4 class-0 correct, 1 of 2 class-1 correct -> (0.75 + 0.5) / 2 under the identity match
    pred = np.asarray([0, 0, 0, 1, 1, 0])
    assert balanced_accuracy_matched(pred, labels) == pytest.approx(0.625)
    with pytest.raises(ConfigError):
        balanced_accuracy_matched(np.arange(7), np.arange(7))
    with pytest.raises(ConfigError):
        balanced_accuracy_matched(labels, labels[:-1])


# --- other predictive metrics --------------------------------------------------------------------


def test_e7_t01_rmse_mae_use_the_forecast_mean_and_declare_their_aggregation() -> None:
    """`rmse`/`mae` per origin are `|mean[h-1] - Y_h|`; `rmse` asks for root-mean-square aggregation."""
    samples = np.random.default_rng(4).normal(0.01, 0.02, size=(4000, 3, 1))
    fc = forecast_of_samples(samples)
    realized = np.asarray([[0.02], [-0.01], [0.03]])
    expected = abs(float(fc.mean[1, 0]) - 0.01)
    assert metric("mae")(forecast=fc, realized=realized, horizon=2) == pytest.approx(expected)
    assert metric("rmse")(forecast=fc, realized=realized, horizon=2) == pytest.approx(expected)
    assert metric("rmse").aggregate == "rms"
    assert metric("mae").aggregate == "mean"


def test_e7_t01_several_series_give_per_series_entries() -> None:
    """`M = 2`: scalar metrics return `<name>_<m>`; `pit_coverage` suffixes its keys."""
    rng = np.random.default_rng(5)
    fc = forecast_of_samples(rng.normal(size=(500, 2, 2)))
    realized = np.asarray([[0.1, -0.2], [0.0, 0.3]])
    out = metric("crps")(forecast=fc, realized=realized, horizon=2)
    assert set(out) == {"crps_0", "crps_1"}  # type: ignore[arg-type]
    cov = metric("pit_coverage")(forecast=fc, realized=realized, horizon=2)
    assert {"pit_0", "pit_1", "coverage_90_0", "coverage_90_1"} <= set(cov)  # type: ignore[arg-type]


def test_e7_t01_forecast_metrics_validate_their_context() -> None:
    """Missing context, a horizon beyond the forecast or realized path, or no samples -> `ConfigError`."""
    fc = forecast_of_samples(np.random.default_rng(6).normal(size=(100, 2, 1)))
    crps = metric("crps")
    with pytest.raises(ConfigError):
        crps(forecast=fc, horizon=1)
    with pytest.raises(ConfigError):
        crps(forecast=fc, realized=np.zeros((2, 1)), horizon=3)
    with pytest.raises(ConfigError):
        crps(forecast=fc, realized=np.zeros((1, 1)), horizon=2)
    empty = ForecastResult(**{**fc.__dict__, "samples": jnp.zeros((0, 2, 1))})
    with pytest.raises(ConfigError):
        crps(forecast=empty, realized=np.zeros((2, 1)), horizon=1)


# --- model metrics -------------------------------------------------------------------------------


def test_e7_t01_model_metrics_read_the_decoded_posterior() -> None:
    """`aic`/`bic`/`icl` mirror the posterior; occupancy sums to 1; persistence = mean segment length."""
    case = make_case("gbm", "negbin", 1, t_len=80, seed=8)
    post = decode(fit_of(case), case.data)
    assert metric("aic")(posterior=post) == post.aic
    assert metric("bic")(posterior=post) == post.bic
    assert metric("icl")(posterior=post) == post.icl

    occupancy = metric("regime_occupancy")(posterior=post)
    assert sum(occupancy.values()) == pytest.approx(1.0)  # type: ignore[union-attr]
    path = np.asarray(post.viterbi)
    for i in range(case.spec.n_states):
        assert occupancy[str(i)] == pytest.approx(float(np.mean(path == i)))  # type: ignore[index]

    persistence = metric("regime_persistence")(posterior=post)
    for regime, mean_len in persistence.items():  # type: ignore[union-attr]
        lengths = [e - s + 1 for s, e, r in post.segments if r == int(regime)]
        assert mean_len == pytest.approx(np.mean(lengths))
    assert sum(e - s + 1 for s, e, _ in post.segments) == path.shape[0]


def test_e7_t01_balanced_accuracy_metric_needs_labels_and_a_posterior() -> None:
    """The registered metric scores the Viterbi path against `labels`, and refuses missing inputs."""
    case = make_case("gbm", "negbin", 1, t_len=60, seed=9)
    post = decode(fit_of(case), case.data)
    m = metric("balanced_accuracy")
    assert m.needs == frozenset({"posterior", "labels"})
    assert m(posterior=post, labels=np.asarray(post.viterbi)) == 1.0
    assert m(posterior=post, labels=1 - np.asarray(post.viterbi)) == 1.0
    with pytest.raises(ConfigError):
        m(posterior=post, labels=None)
    with pytest.raises(ConfigError):
        m(labels=np.zeros(60, dtype=int))


def test_e7_t01_log_score_mixture_with_exact_weights_matches_an_independent_computation() -> None:
    """Exact `(softmax(alpha_T) A_{T+1}) C` weights vs a NumPy recomputation; the MC-weight path is close."""
    case = make_case("gbm", "negbin", 1, t_len=60, seed=11)
    fit = fit_of(case)
    log_alpha, _ = case.engine.forward(fit.params, case.spec, case.space, case.data)
    exact = np.asarray(one_step_mixture_weights(log_alpha[-1], fit.params, case.space, DT))

    _, log_a = case.space.transition_fn(fit.params, jnp.zeros((0,)), jnp.zeros((0,)), jnp.asarray(DT))
    alpha = np.exp(np.asarray(log_alpha[-1]) - np.max(np.asarray(log_alpha[-1])))
    alpha /= alpha.sum()
    reference = alpha @ np.exp(np.asarray(log_a)) @ np.asarray(case.space.index.collapse)
    np.testing.assert_allclose(exact, reference, atol=1e-12)

    y = 0.003
    mu = np.asarray(fit.params["emission.mean"]).reshape(-1)
    sigma = np.asarray(fit.params["emission.scale"]).reshape(-1)
    density = np.sum(reference * stats.norm.pdf(y, (mu - 0.5 * sigma**2) * DT, sigma * np.sqrt(DT)))
    cfg = ForecastSection(
        horizon=1,
        n_sim=20000,
        covariate_forecaster=CovariateForecasterCfg(type="freeze"),
        quantiles=[0.05, 0.5, 0.95],
    )
    mc = MonteCarloForecaster(case.spec, case.space, case.engine, None, cfg)
    fc = mc.forecast(fit, case.data, 1, 20000, seed=0)
    ctx = {"forecast": fc, "fit": fit, "realized": np.asarray([[y]]), "dt_next": DT, "horizon": 1}
    with_exact = metric("log_score_mixture")(**ctx, mixture_weights=exact)
    assert with_exact == pytest.approx(float(np.log(density)), abs=1e-8)
    assert abs(metric("log_score_mixture")(**ctx) - with_exact) < 0.01  # MC weights: within MC error


def test_e7_t01_log_score_mixture_is_horizon_one_only() -> None:
    """Asking for any other horizon is a `ConfigError`, not a silent step-1 score."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=3)
    fit = fit_of(case)
    fc = forecast_of_samples(np.zeros((4, 2, 1)))
    with pytest.raises(ConfigError):
        metric("log_score_mixture")(forecast=fc, fit=fit, realized=np.zeros((2, 1)), dt_next=DT, horizon=2)


def test_e7_t01_custom_registration_before_builtin_loading_does_not_clash() -> None:
    """In a fresh interpreter, replacing a builtin before any `resolve` still works and sticks."""
    code = (
        "from regimekit.evaluation.metrics.registry import FunctionMetric, register, resolve\n"
        "register(FunctionMetric('mae', frozenset({'forecast'}), lambda **_: 7.0), replace=True)\n"
        "assert resolve(['mae', 'crps'])[0]() == 7.0\n"
        "try:\n"
        "    register(FunctionMetric('crps', frozenset(), lambda **_: 0.0))\n"
        "except Exception as exc:\n"
        "    assert 'already registered' in str(exc)\n"
        "else:\n"
        "    raise SystemExit('duplicate accepted')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
