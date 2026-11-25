"""E7-T02b: walk-forward runner -- fold table, metric rows, summary, standardization, padding, seeds."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

from regimekit.config.schema import ProjectConfig
from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.data.prepare import Preparer
from regimekit.engine import expanded_hmm
from regimekit.estimation import objective
from regimekit.evaluation import backtest
from regimekit.evaluation.backtest import run_backtest, schedule_folds
from regimekit.utils.errors import ConfigError
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

ALL_METRICS = (
    "log_score",
    "log_score_mixture",
    "crps",
    "pit_coverage",
    "rmse",
    "mae",
    "bic",
    "regime_occupancy",
    "sharpe",  # needs a strategy trace: skipped here (E7-T03)
)
# A short, cheap run: 200 prices, two folds, one horizon.
SHORT = {"initial_train": 100, "refit_every": 30, "horizons": (1,), "restarts": 1, "n_sim": 50}


def _short_inputs(tmp_path: Path, **overrides: object) -> tuple[ProjectConfig, ObservationSet]:
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    cfg = bf.make_config(tmp_path, metrics=("log_score", "crps"), **{**SHORT, **overrides})  # type: ignore[arg-type]
    return cfg, obs


def test_e7_t02b_fold_table_metric_rows_summary_and_single_compilation(tmp_path: Path) -> None:
    """AC2 (+ AC4 compile count): fold table == `schedule_folds`; rows per (fold, horizon, metric)."""
    jax.clear_caches()
    returns = bf.simulate_returns(bf.N_PRICES - 1)
    obs = bf.observations(returns)
    cfg = bf.make_config(tmp_path, metrics=ALL_METRICS)
    result = run_backtest(cfg, lambda: (obs, []), seed=5)

    # --- fold table equals the scheduler's output
    grid = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    expected = schedule_folds(
        np.asarray(grid.times), np.asarray(grid.dt), cfg.evaluation.backtest, max_horizon=5
    )
    cols = ["fold", "train_start", "train_end", "test_start", "test_end"]
    assert result.folds[cols].to_numpy().tolist() == [
        [f.fold, f.train_start, f.train_end, f.test_start, f.test_end] for f in expected
    ]
    assert (result.folds["fit_time"] > 0).all()
    assert result.extras == {}

    # --- metrics: one row per (fold, horizon, metric); horizon 0 = per-fold model metrics
    m = result.metrics
    assert list(m.columns) == ["fold", "horizon", "metric", "value"]
    assert not m.duplicated(["fold", "horizon", "metric"]).any()
    assert np.isfinite(m["value"]).all()
    names_h1 = {
        "log_score",
        "log_score_mixture",
        "crps",
        "rmse",
        "mae",
        "pit_coverage.pit",
        "pit_coverage.coverage_50",
        "pit_coverage.coverage_90",
    }
    for fold in range(len(expected)):
        rows = m[m["fold"] == fold]
        assert set(rows[rows["horizon"] == 1]["metric"]) == names_h1
        assert set(rows[rows["horizon"] == 5]["metric"]) == names_h1 - {"log_score_mixture"}
        assert set(rows[rows["horizon"] == 0]["metric"]) == {
            "bic",
            "regime_occupancy.0",
            "regime_occupancy.1",
        }
    assert "sharpe" not in set(m["metric"])

    # --- summary: one row per (horizon, metric); mean / sd / n across folds, rmse pooled as RMS
    s = result.summary.set_index(["horizon", "metric"])
    assert not s.index.duplicated().any()
    for (horizon, name), row in s.iterrows():
        if name.startswith("pit_ks"):
            continue
        vals = m[(m["horizon"] == horizon) & (m["metric"] == name)]["value"].to_numpy()
        assert row["n"] == len(expected) == vals.shape[0]
        want = float(np.sqrt(np.mean(vals**2))) if name == "rmse" else float(np.mean(vals))
        assert row["mean"] == pytest.approx(want, rel=1e-12), (horizon, name)
        assert row["sd"] == pytest.approx(np.std(vals, ddof=1), rel=1e-12)
    n_origins = int(sum(f.test_end - f.test_start for f in expected))
    for horizon in (1, 5):
        assert s.loc[(horizon, "pit_ks"), "n"] == n_origins
        assert 0.0 <= s.loc[(horizon, "pit_ks"), "mean"] <= 1.0
        assert 0.0 <= s.loc[(horizon, "pit_ks_pvalue"), "mean"] <= 1.0
    # honest forecasts of the true model class: 90 % intervals cover close to 90 %
    assert s.loc[(1, "pit_coverage.coverage_90"), "mean"] == pytest.approx(0.9, abs=0.1)
    # the exact mixture score and the KDE score agree on a one-step forecast
    assert s.loc[(1, "log_score_mixture"), "mean"] == pytest.approx(s.loc[(1, "log_score"), "mean"], abs=0.1)

    # --- AC4: 512-row windows for every fold -> the fit and the filter compile once, not once per fold
    assert objective._value_and_grad._cache_size() == 1
    assert expanded_hmm._forward_core._cache_size() == 1


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_e7_t02b_standardization_uses_only_each_folds_train_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    """AC3: per-fold `PrepStats.train_window` ends at the fold's last train row; moments are train-only."""
    calls: list[tuple[pd.Timestamp, object]] = []

    class SpyPreparer(Preparer):
        def fit(
            self,
            obs: ObservationSet,
            covs: Sequence[CovariateSeries],
            train_end: str | np.datetime64 | pd.Timestamp,
            train_start: str | np.datetime64 | pd.Timestamp | None = None,
        ) -> Preparer:
            out = super().fit(obs, covs, train_end, train_start)
            calls.append((pd.Timestamp(train_end), self.prep_stats))
            return out

    monkeypatch.setattr(backtest, "Preparer", SpyPreparer)
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    cov = bf.covariate(obs, np.concatenate([[0.0], returns]))
    cfg = bf.make_config(  # type: ignore[arg-type]
        tmp_path, with_covariate=True, metrics=("log_score",), scheme=scheme, **SHORT
    )
    result = run_backtest(cfg, lambda: (obs, [cov]), seed=5)

    fold_calls = calls[1:]  # the first call prepares the full-sample grid (shape only)
    assert len(fold_calls) == len(result.folds) == 4
    # independent expectation: the unstandardized covariate block of the full grid
    raw_cfg = cfg.model_copy(deep=True)
    raw_cfg.data.preparation.standardize_covariates = False
    raw = Preparer(raw_cfg.data).fit(obs, [cov], train_end=obs.times[-1]).transform(obs, [cov])
    z = np.asarray(raw.cov_emission)[:, 0]
    full_mean, full_std = z.mean(), z.std()
    means = []
    for (train_end, stats), (_, row) in zip(fold_calls, result.folds.iterrows(), strict=True):
        n_train, n_start = int(row["train_end"]), int(row["train_start"])
        assert train_end == pd.Timestamp(row["train_end_time"])
        assert pd.Timestamp(stats.train_window[0]) == pd.Timestamp(row["train_start_time"])
        assert pd.Timestamp(stats.train_window[1]) == pd.Timestamp(row["train_end_time"])  # type: ignore[attr-defined]
        np.testing.assert_allclose(stats.cov_mean, [z[n_start:n_train].mean()], rtol=1e-12, atol=1e-15)  # type: ignore[attr-defined]
        np.testing.assert_allclose(stats.cov_std, [z[n_start:n_train].std()], rtol=1e-12)  # type: ignore[attr-defined]
        means.append(float(stats.cov_mean[0]))  # type: ignore[attr-defined]
    assert means[0] != means[1]  # the window grows, the statistics move
    assert not np.isclose(means[0], full_mean, rtol=1e-6) or not np.isclose(
        fold_calls[0][1].cov_std[0],  # type: ignore[attr-defined]
        full_std,
        rtol=1e-6,
    )


def test_e7_t02b_padding_gives_the_same_results_as_no_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4: padded windows (masked rows, 256 multiple) reproduce the unpadded run to 1e-8."""
    cfg, obs = _short_inputs(tmp_path, sharp=True)
    padded = run_backtest(cfg, lambda: (obs, []), seed=5)
    monkeypatch.setattr(backtest, "pad_dataset", lambda data, *a, **k: data)
    plain = run_backtest(cfg, lambda: (obs, []), seed=5)
    assert padded.metrics[["fold", "horizon", "metric"]].equals(plain.metrics[["fold", "horizon", "metric"]])
    np.testing.assert_allclose(
        padded.metrics["value"].to_numpy(), plain.metrics["value"].to_numpy(), rtol=0, atol=1e-8
    )


def test_e7_t02b_same_seed_gives_identical_metrics_and_a_new_seed_differs(tmp_path: Path) -> None:
    """AC5: two runs with the same seed give identical `metrics` (bitwise); another seed does not."""
    cfg, obs = _short_inputs(tmp_path)
    first = run_backtest(cfg, lambda: (obs, []), seed=5)
    again = run_backtest(cfg, lambda: (obs, []), seed=5)
    other = run_backtest(cfg, lambda: (obs, []), seed=6)
    pd.testing.assert_frame_equal(first.metrics, again.metrics, check_exact=True)
    pd.testing.assert_frame_equal(first.summary, again.summary, check_exact=True)
    assert not np.array_equal(first.metrics["value"].to_numpy(), other.metrics["value"].to_numpy())


def test_e7_t02b_scenario_covariate_forecaster_is_a_config_error(tmp_path: Path) -> None:
    """One fixed scenario cannot serve every origin: `ConfigError` naming the config path."""
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    cov = bf.covariate(obs, np.concatenate([[0.0], returns]))
    cfg = bf.make_config(tmp_path, with_covariate=True, metrics=("log_score",), **SHORT)  # type: ignore[arg-type]
    cfg.forecast.covariate_forecaster.type = "scenario"
    with pytest.raises(ConfigError, match="scenario") as err:
        run_backtest(cfg, lambda: (obs, [cov]), seed=5)
    assert err.value.path == "forecast.covariate_forecaster.type"
