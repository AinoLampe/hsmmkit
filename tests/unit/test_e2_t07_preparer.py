"""E2-T07: `Preparer` fit/transform, transforms, train-only standardization, missing policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regimekit.config.schema import SourceCfg
from regimekit.config.yaml_io import load_config
from regimekit.data.contracts import CovariateSeries, ObservationSet, PreparedDataset, PrepStats
from regimekit.data.prepare import Preparer, prep_stats_from_json, prep_stats_to_json
from regimekit.data.transforms import apply_transforms, log_returns
from regimekit.finance import FinanceConfig
from regimekit.finance.config import FinanceCovariateCfg, FinanceDataSection, FinancePrepCfg
from regimekit.utils.errors import ConfigError, DataError

N_DAYS = 200
DATES = pd.bdate_range("2021-01-04", periods=N_DAYS)
TIMES = np.asarray(DATES.as_unit("ns").values, dtype="datetime64[ns]")


def _prices(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, N_DAYS)))


def _obs(
    prices: np.ndarray | None = None, times: np.ndarray = TIMES, snapshot_id: str = "p-snap"
) -> ObservationSet:
    p = _prices() if prices is None else prices
    return ObservationSet(
        symbols=("X",),
        times=times,
        prices=np.asarray(p, dtype=np.float64).reshape(len(times), -1),
        interval="1d",
        adjusted=True,
        provider="csv",
        snapshot_id=snapshot_id,
    )


def _cov(series_id: str, values: np.ndarray, times: np.ndarray = TIMES) -> CovariateSeries:
    return CovariateSeries(
        name=series_id,
        times=times,
        values=np.asarray(values, dtype=np.float64),
        provider="csv",
        series_id=series_id,
        snapshot_id=f"{series_id}-snap",
        frequency="d",
    )


def _trend(seed: int = 1) -> np.ndarray:
    """A strictly positive, trending covariate (so full-sample standardized mean != 0)."""
    rng = np.random.default_rng(seed)
    return 20.0 + np.linspace(0.0, 30.0, N_DAYS) + rng.normal(0.0, 1.0, N_DAYS)


def _v2_cov(c: dict[str, Any]) -> dict[str, Any]:
    """A Part I covariate dict (`provider`, `series`) in the version-2 layout (`source`)."""
    c = dict(c)
    c["source"] = {"kind": "provider", "provider": c.pop("provider"), "series": c.pop("series")}
    return c


def _cfg(
    covariates: list[dict[str, Any]] | None = None,
    *,
    clock: str = "trading",
    standardize: bool = True,
    missing_returns: str = "drop",
) -> FinanceDataSection:
    return FinanceDataSection(
        source=SourceCfg(kind="provider", provider="csv", symbols=["X"], interval="1d", start="2021-01-04"),
        clock=clock,  # type: ignore[arg-type]
        covariates=[FinanceCovariateCfg(**_v2_cov(c)) for c in (covariates or [])],
        preparation=FinancePrepCfg(standardize_covariates=standardize, missing_observations=missing_returns),  # type: ignore[arg-type]
    )


def _covcfg(name: str, series: str, **kw: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "provider": "csv",
        "series": series,
        "lag": 1,
        "use_in": ["transition"],
    }
    base.update(kw)
    return base


TRAIN_END = str(DATES[int(0.6 * N_DAYS)].date())


# --- observations ------------------------------------------------------------------------------


def test_e2_t07_log_returns_pure_function() -> None:
    """`log_returns` is `log(P_t / P_{t-1})` per series, shape (T, M)."""
    prices = np.array([[100.0, 10.0], [110.0, 5.0], [99.0, 20.0]])
    y = log_returns(prices)
    np.testing.assert_allclose(np.asarray(y), np.log(prices[1:] / prices[:-1]), rtol=0, atol=1e-15)
    assert y.shape == (2, 2)


def test_e2_t07_observations_mask_and_fields() -> None:
    """Returns match the prices, non-finite returns are masked and zeroed, Part II fields are defaults."""
    prices = _prices()
    prices[50] = 0.0  # log(0) -> -inf return at rows 49 (into) and 50 (out of the zero)
    data = Preparer(_cfg()).fit(_obs(prices), [], TRAIN_END).transform(_obs(prices), [])
    y = np.asarray(data.observations)
    mask = np.asarray(data.mask)
    assert y.shape == (N_DAYS - 1, 1)
    assert mask.dtype == np.bool_
    assert not mask[49, 0]
    assert not mask[50, 0]
    assert y[49, 0] == 0.0
    assert y[50, 0] == 0.0
    ok = np.ones(N_DAYS - 1, dtype=bool)
    ok[[49, 50]] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.log(prices[1:] / prices[:-1])
    np.testing.assert_allclose(y[ok, 0], expected[ok], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(data.times, TIMES[1:])
    assert data.support == "real"
    assert data.n_categories is None
    assert data.labels is None
    assert data.boundaries is None
    assert data.group is None
    assert data.clock == "trading"
    assert data.snapshot_ids == ("p-snap",)
    assert data.cov_transition is None
    assert data.cov_duration is None
    assert data.cov_emission is None
    data.validate()


def test_e2_t07_missing_price_error_policy() -> None:
    """`missing_returns: error` raises on a missing primary price."""
    prices = _prices()
    prices[10] = np.nan
    with pytest.raises(DataError, match=str(DATES[10].date())):
        Preparer(_cfg(missing_returns="error")).fit(_obs(prices), [], TRAIN_END)


# --- AC1: train-only standardization -----------------------------------------------------------


def test_e2_t07_transform_uses_train_window_stats() -> None:
    """AC1: fit on the first 60 %; train rows are standardized to mean 0 / std 1, the full series is not."""
    cfg = _cfg([_covcfg("z", "Z")])
    obs, covs = _obs(), [_cov("Z", _trend())]
    prep = Preparer(cfg).fit(obs, covs, TRAIN_END)
    data = prep.transform(obs, covs)
    z = np.asarray(data.cov_transition)[:, 0]
    train = data.times <= np.datetime64(TRAIN_END, "ns")
    assert 100 < train.sum() < N_DAYS - 50
    assert abs(z[train].mean()) < 1e-12
    assert abs(z[train].std() - 1.0) < 1e-12
    assert z.mean() > 0.5  # trending covariate: the test part sits well above the train mean
    # Stored stats are exactly the train-row moments of the lagged raw covariate.
    raw = _trend()[:-1]  # lag 1: row t holds the print dated τ_t
    np.testing.assert_allclose(prep.prep_stats.cov_mean, [raw[train].mean()], rtol=1e-14)
    np.testing.assert_allclose(prep.prep_stats.cov_std, [raw[train].std()], rtol=1e-14)
    assert prep.prep_stats.train_window == (str(DATES[1].date()), TRAIN_END)


def test_e2_t07_transform_on_test_window_reuses_train_stats() -> None:
    """Transforming a later window alone gives the same values as the full transform there."""
    cfg = _cfg([_covcfg("z", "Z")])
    obs, covs = _obs(), [_cov("Z", _trend())]
    prep = Preparer(cfg).fit(obs, covs, TRAIN_END)
    full = prep.transform(obs, covs)
    cut = 120
    late = prep.transform(_obs(_prices()[cut:], TIMES[cut:]), [_cov("Z", _trend()[cut:], TIMES[cut:])])
    offset = len(full.times) - len(late.times)
    np.testing.assert_array_equal(late.times, full.times[offset:])
    np.testing.assert_allclose(
        np.asarray(late.cov_transition), np.asarray(full.cov_transition)[offset:], rtol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(late.observations), np.asarray(full.observations)[offset:], rtol=1e-15
    )


def test_e2_t07_no_standardization_keeps_raw_values() -> None:
    """`standardize_covariates: false` stores identity stats and leaves values raw."""
    cfg = _cfg([_covcfg("z", "Z")], standardize=False)
    obs, covs = _obs(), [_cov("Z", _trend())]
    data = Preparer(cfg).fit(obs, covs, TRAIN_END).transform(obs, covs)
    np.testing.assert_array_equal(np.asarray(data.cov_transition)[:, 0], _trend()[:-1])
    np.testing.assert_array_equal(data.prep_stats.cov_mean, [0.0])
    np.testing.assert_array_equal(data.prep_stats.cov_std, [1.0])


def test_e2_t07_zscore_is_a_deprecated_standardization_alias() -> None:
    """`zscore` standardizes with train stats (not full-sample) even when standardization is off."""
    cfg = _cfg([_covcfg("z", "Z", transform=["zscore"])], standardize=False)
    obs, covs = _obs(), [_cov("Z", _trend())]
    with pytest.warns(FutureWarning, match="zscore"):
        prep = Preparer(cfg).fit(obs, covs, TRAIN_END)
    data = prep.transform(obs, covs)
    z = np.asarray(data.cov_transition)[:, 0]
    train = data.times <= np.datetime64(TRAIN_END, "ns")
    assert abs(z[train].mean()) < 1e-12
    assert z.mean() > 0.5


def test_e2_t07_constant_covariate_raises() -> None:
    """A train-window std below 1e-12 is a DataError."""
    cfg = _cfg([_covcfg("z", "Z")])
    with pytest.raises(DataError, match="std") as info:
        Preparer(cfg).fit(_obs(), [_cov("Z", np.full(N_DAYS, 3.0))], TRAIN_END)
    assert info.value.path == "data.covariates[0]"


def test_e2_t07_transform_before_fit_raises() -> None:
    """`transform` without `fit` is an error; so is `prep_stats`."""
    prep = Preparer(_cfg())
    with pytest.raises(DataError, match="fit"):
        prep.transform(_obs(), [])
    with pytest.raises(DataError, match="fit"):
        _ = prep.prep_stats


def test_e2_t07_train_end_before_data_raises() -> None:
    """A `train_end` before the first observation leaves no training rows."""
    with pytest.raises(DataError, match="train_end"):
        Preparer(_cfg([_covcfg("z", "Z")])).fit(_obs(), [_cov("Z", _trend())], "2000-01-01")


def test_e2_t07_covariates_must_match_config() -> None:
    """Covariates pair with `data.covariates[i]` by position and series id."""
    cfg = _cfg([_covcfg("z", "Z")])
    with pytest.raises(DataError, match="expected 1"):
        Preparer(cfg).fit(_obs(), [], TRAIN_END)
    with pytest.raises(DataError, match="'Z'") as info:
        Preparer(cfg).fit(_obs(), [_cov("OTHER", _trend())], TRAIN_END)
    assert info.value.path == "data.covariates[0].series"


# --- AC2: missing policies ---------------------------------------------------------------------


def _holey(start: int, length: int) -> np.ndarray:
    values = _trend()
    values[start : start + length] = np.nan
    return values


def test_e2_t07_ffill_carries_up_to_max_staleness() -> None:
    """AC2 `ffill`: a 3-day hole within `max_staleness=5` is filled with the last print; T unchanged."""
    cfg = _cfg([_covcfg("z", "Z", missing="ffill", max_staleness=5)], standardize=False)
    covs = [_cov("Z", _holey(100, 3))]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    z = np.asarray(data.cov_transition)[:, 0]
    assert z.shape == (N_DAYS - 1,)
    # Row t holds the print dated τ_t; prints 100..102 are missing -> rows 100..102 carry print 99.
    np.testing.assert_array_equal(z[100:103], _trend()[99])
    assert z[103] == _trend()[103]


def test_e2_t07_ffill_drops_rows_beyond_max_staleness() -> None:
    """AC2 `ffill`: rows still NaN after `max_staleness` are dropped jointly (logged)."""
    cfg = _cfg([_covcfg("z", "Z", missing="ffill", max_staleness=2)], standardize=False)
    covs = [_cov("Z", _holey(100, 5))]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    # Rows 100, 101 carry print 99 (staleness 1, 2); rows 102..104 exceed it and are dropped.
    _assert_aligned(data, N_DAYS - 1 - 3)
    dropped = TIMES[1:][102:105]
    assert not np.isin(dropped, data.times).any()


def test_e2_t07_drop_keeps_everything_aligned() -> None:
    """AC2 `drop`: dropped rows vanish from observations, dt, mask, times and every covariate block."""
    cfg = _cfg(
        [
            _covcfg("a", "A", missing="drop", max_staleness=0, use_in=["transition", "emission"]),
            _covcfg("b", "B", missing="ffill", use_in=["duration"]),
        ],
        standardize=False,
        clock="calendar",
    )
    obs = _obs()
    covs = [_cov("A", _holey(60, 4)), _cov("B", _trend(seed=2))]
    full_cfg = _cfg(
        [_covcfg("a", "A", use_in=["transition", "emission"]), _covcfg("b", "B", use_in=["duration"])],
        standardize=False,
        clock="calendar",
    )
    reference = Preparer(full_cfg).fit(obs, [_cov("A", _trend()), covs[1]], TRAIN_END)
    ref = reference.transform(obs, [_cov("A", _trend()), covs[1]])
    data = Preparer(cfg).fit(obs, covs, TRAIN_END).transform(obs, covs)
    _assert_aligned(data, N_DAYS - 1 - 4)
    keep = np.isin(ref.times, data.times)
    assert (~keep).sum() == 4
    np.testing.assert_array_equal(np.asarray(data.observations), np.asarray(ref.observations)[keep])
    np.testing.assert_array_equal(np.asarray(data.dt), np.asarray(ref.dt)[keep])
    np.testing.assert_array_equal(np.asarray(data.cov_duration), np.asarray(ref.cov_duration)[keep])
    np.testing.assert_array_equal(np.asarray(data.cov_emission), np.asarray(ref.cov_emission)[keep])
    data.validate()


def test_e2_t07_error_policy_names_first_timestamp() -> None:
    """AC2 `error`: a gap beyond `max_staleness` raises with the first offending timestamp."""
    cfg = _cfg([_covcfg("z", "Z", missing="error", max_staleness=1)])
    covs = [_cov("Z", _holey(100, 4))]
    # Row 101 (observation ending τ_102) is the first with staleness 2.
    with pytest.raises(DataError, match=str(DATES[102].date())) as info:
        Preparer(cfg).fit(_obs(), covs, TRAIN_END)
    assert info.value.path == "data.covariates[0].missing"


def test_e2_t07_error_policy_passes_within_staleness() -> None:
    """`error` does not fire while prints are within `max_staleness`."""
    cfg = _cfg([_covcfg("z", "Z", missing="error", max_staleness=5)])
    covs = [_cov("Z", _holey(100, 4))]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    _assert_aligned(data, N_DAYS - 1)


def test_e2_t07_structural_warmup_is_trimmed_for_every_policy() -> None:
    """Rows lost to the lag shift and to `diff` are trimmed under every policy, `error` included."""
    for missing in ("ffill", "drop", "error"):
        cfg = _cfg(
            [
                _covcfg("a", "A", lag=3, missing=missing),
                _covcfg("b", "B", transform=["log", "diff", "diff"], missing=missing),
            ],
            standardize=False,
        )
        covs = [_cov("A", _trend()), _cov("B", _trend(seed=2))]
        data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
        # A (lag 3) first fills row 2; B (two diffs, lag 1) first fills row 2 as well.
        _assert_aligned(data, N_DAYS - 1 - 2)
        assert data.times[0] == TIMES[3]


def test_e2_t07_late_starting_covariate() -> None:
    """A series starting after the primary is trimmed with a warning, but raises under `error`."""
    late = _cov("B", _trend()[10:], TIMES[10:])
    for missing in ("ffill", "drop"):
        cfg = _cfg([_covcfg("b", "B", transform=["diff"], missing=missing)], standardize=False)
        data = Preparer(cfg).fit(_obs(), [late], TRAIN_END).transform(_obs(), [late])
        # B starts at τ_10 and its first diff exists at τ_11 -> first usable row is 11 (lag 1).
        _assert_aligned(data, N_DAYS - 1 - 11)
        assert data.times[0] == TIMES[12]
    cfg = _cfg([_covcfg("b", "B", transform=["diff"], missing="error")], standardize=False)
    with pytest.raises(DataError, match="missing='error'") as info:
        Preparer(cfg).fit(_obs(), [late], TRAIN_END)
    assert info.value.path == "data.covariates[0].missing"


def _assert_aligned(data: PreparedDataset, t: int) -> None:
    assert data.times.shape == (t,)
    for arr in (data.observations, data.mask):
        assert arr.shape == (t, 1)
    assert data.dt.shape == (t,)
    for block in (data.cov_transition, data.cov_duration, data.cov_emission):
        if block is not None:
            assert block.shape[0] == t
            assert np.all(np.isfinite(np.asarray(block)))
    data.validate()


# --- AC3: transform order vs pandas ------------------------------------------------------------


def test_e2_t07_log_then_diff_matches_pandas() -> None:
    """AC3: `[log, diff]` equals pandas `np.log(s).diff()` on the series' own prints."""
    values = _trend()
    expected = np.log(pd.Series(values)).diff().to_numpy()
    np.testing.assert_allclose(
        apply_transforms(values, ["log", "diff"]), expected, rtol=0, atol=1e-15, equal_nan=True
    )
    cfg = _cfg([_covcfg("z", "Z", transform=["log", "diff"])], standardize=False)
    covs = [_cov("Z", values)]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    # Row t holds the transformed print dated τ_t; row 0 (print 0, diff NaN) is warm-up.
    np.testing.assert_allclose(np.asarray(data.cov_transition)[:, 0], expected[1:-1], rtol=0, atol=1e-15)


def test_e2_t07_transform_order_matters() -> None:
    """`[diff, log]` is not `[log, diff]`; `pct` matches pandas `pct_change`."""
    values = np.cumsum(1.0 + np.random.default_rng(4).uniform(size=N_DAYS))  # increasing: diffs > 0
    swapped = apply_transforms(values, ["diff", "log"])
    np.testing.assert_allclose(
        swapped, np.log(pd.Series(values).diff()).to_numpy(), rtol=1e-15, equal_nan=True
    )
    assert not np.allclose(swapped[1:], apply_transforms(values, ["log", "diff"])[1:])
    np.testing.assert_allclose(
        apply_transforms(values, ["pct"]),
        pd.Series(values).pct_change().to_numpy(),
        rtol=1e-15,
        equal_nan=True,
    )
    np.testing.assert_array_equal(apply_transforms(values, ["zscore"]), values)  # handled by standardization
    np.testing.assert_array_equal(apply_transforms(values, []), values)


def test_e2_t07_transforms_skip_nan_prints() -> None:
    """`diff` is taken between consecutive non-NaN prints (a holiday NaN does not poison two rows)."""
    values = np.array([1.0, 2.0, np.nan, 4.0, 7.0])
    np.testing.assert_array_equal(apply_transforms(values, ["diff"]), [np.nan, 1.0, np.nan, 2.0, 3.0])


def test_e2_t07_log_of_non_positive_raises() -> None:
    """`log` of a non-positive print is a DataError rather than a silent NaN."""
    with pytest.raises(DataError, match="log"):
        apply_transforms(np.array([1.0, 0.0, 2.0]), ["log"])


# --- AC4: PrepStats JSON, validate() incl. steps clock -----------------------------------------


def test_e2_t07_prep_stats_json_round_trip() -> None:
    """AC4: `PrepStats` survives a JSON round trip bit-for-bit."""
    cfg = _cfg([_covcfg("a", "A", transform=["log"]), _covcfg("b", "B", lag=2, use_in=["duration"])])
    covs = [_cov("A", _trend()), _cov("B", _trend(seed=3))]
    stats = Preparer(cfg).fit(_obs(), covs, TRAIN_END).prep_stats
    text = prep_stats_to_json(stats)
    assert json.loads(text)["cov_names"] == ["a", "b"]
    back = prep_stats_from_json(text)
    assert isinstance(back, PrepStats)
    assert back == stats
    assert hash(back) == hash(stats)
    assert back.lags == {"a": 1, "b": 2}
    assert back.transforms == {"a": ["log"], "b": []}


def test_e2_t07_validate_passes_on_appendix_a() -> None:
    """AC4: the Appendix A data section yields a valid dataset (two FRED-like covariates)."""
    cfg = load_config(
        Path(__file__).parents[1] / "fixtures" / "configs" / "appendix_a.yaml", FinanceConfig
    ).data
    vix = _trend()
    vix[[30, 31]] = np.nan  # FRED "." holidays
    covs = [_cov("VIXCLS", vix), _cov("T10Y2Y", np.sin(np.arange(N_DAYS) / 10.0))]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    data.validate()
    assert data.cov_names == {"transition": ("log_vix", "term_spread"), "duration": ("log_vix",)}
    assert data.snapshot_ids == ("p-snap", "VIXCLS-snap", "T10Y2Y-snap")
    assert np.all(np.asarray(data.dt) == 1.0 / 252.0)
    assert data.nominal_dt == 1.0 / 252.0


def test_e2_t07_validate_passes_under_steps_clock() -> None:
    """AC4: under `clock: steps` every `dt` is exactly 1 and `validate()` passes."""
    cfg = _cfg([_covcfg("z", "Z")], clock="steps")
    covs = [_cov("Z", _trend())]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    data.validate()
    assert data.clock == "steps"
    assert data.nominal_dt == 1.0
    np.testing.assert_array_equal(np.asarray(data.dt), 1.0)


# --- AC5: entry-point split --------------------------------------------------------------------


def test_e2_t07_covariate_in_two_entry_points_has_identical_columns() -> None:
    """AC5: `use_in: [transition, duration]` puts the same column in both blocks."""
    cfg = _cfg(
        [
            _covcfg("a", "A", use_in=["transition", "duration"]),
            _covcfg("b", "B", use_in=["transition"]),
            _covcfg("c", "C", use_in=["emission", "duration"]),
        ]
    )
    covs = [_cov("A", _trend()), _cov("B", _trend(seed=2)), _cov("C", _trend(seed=3))]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    assert data.cov_names == {
        "transition": ("a", "b"),
        "duration": ("a", "c"),
        "emission": ("c",),
    }
    tr, du, em = (np.asarray(b) for b in (data.cov_transition, data.cov_duration, data.cov_emission))
    np.testing.assert_array_equal(tr[:, 0], du[:, 0])
    np.testing.assert_array_equal(du[:, 1], em[:, 0])
    assert not np.array_equal(tr[:, 0], tr[:, 1])


def test_e2_t07_config_error_for_zero_lag_propagates() -> None:
    """A programmatically built lag of 0 (bypassing the schema) still raises ConfigError."""
    valid = _cfg([_covcfg("z", "Z")])
    cfg = valid.model_copy(update={"covariates": [valid.covariates[0].model_copy(update={"lag": 0})]})
    with pytest.raises(ConfigError) as info:
        Preparer(cfg).fit(_obs(), [_cov("Z", _trend())], TRAIN_END)
    assert info.value.path == "data.covariates[0].lag"


def test_e2_t07_pipeline_error_paths() -> None:
    """Unusable covariate, unknown transform and a config changed after `fit` are DataErrors."""
    cfg = _cfg([_covcfg("z", "Z")])
    with pytest.raises(DataError, match="no usable value") as info:
        Preparer(cfg).fit(_obs(), [_cov("Z", np.full(N_DAYS, np.nan))], TRAIN_END)
    assert info.value.path == "data.covariates[0]"
    with pytest.raises(DataError, match="unknown transform") as info:
        apply_transforms(np.ones(3), ["sqrt"], path="data.covariates[0].transform")
    assert info.value.path == "data.covariates[0].transform"
    prep = Preparer(cfg).fit(_obs(), [_cov("Z", _trend())], TRAIN_END)
    prep.cfg = _cfg([_covcfg("other", "Z")])
    with pytest.raises(DataError, match="differ from the config"):
        prep.transform(_obs(), [_cov("Z", _trend())])


# --- review follow-ups: no look-ahead into stats, non-finite values, time handling ----------------


def test_e2_t07_stats_ignore_data_after_train_end() -> None:
    """Changing prices and covariates after `train_end` leaves `PrepStats` and train rows bit-identical."""
    cfg = _cfg([_covcfg("a", "A", transform=["log", "diff"]), _covcfg("b", "B", lag=2)])
    covs = [_cov("A", _trend()), _cov("B", _trend(seed=2))]
    base = Preparer(cfg).fit(_obs(), covs, TRAIN_END)
    cut = int(0.6 * N_DAYS) + 1  # first print dated after train_end
    prices = _prices()
    prices[cut:] *= np.linspace(1.0, 3.0, N_DAYS - cut)
    a, b = _trend(), _trend(seed=2)
    a[cut:] += 1000.0
    b[cut:] = -b[cut:]
    changed_covs = [_cov("A", a), _cov("B", b)]
    changed = Preparer(cfg).fit(_obs(prices), changed_covs, TRAIN_END)
    assert changed.prep_stats == base.prep_stats
    d0, d1 = base.transform(_obs(), covs), changed.transform(_obs(prices), changed_covs)
    train = d0.times <= np.datetime64(TRAIN_END, "ns")
    np.testing.assert_array_equal(np.asarray(d1.observations)[train], np.asarray(d0.observations)[train])
    np.testing.assert_array_equal(np.asarray(d1.cov_transition)[train], np.asarray(d0.cov_transition)[train])


def test_e2_t07_timezone_aware_train_end_is_converted_to_utc() -> None:
    """`train_end` with an offset is compared in UTC, not by its wall-clock time."""
    cfg = _cfg([_covcfg("z", "Z")])
    covs = [_cov("Z", _trend())]
    day = DATES[120]
    # 00:00+02:00 on `day` is 22:00 UTC the day before: the row ending at `day` is not in training.
    before = Preparer(cfg).fit(_obs(), covs, f"{day.date()}T00:00:00+02:00").prep_stats
    assert before.train_window[1] == str(DATES[119].date())
    after = Preparer(cfg).fit(_obs(), covs, f"{day.date()}T02:00:00+02:00").prep_stats
    assert after.train_window[1] == str(day.date())
    naive = Preparer(cfg).fit(_obs(), covs, np.datetime64(day.date(), "ns")).prep_stats
    assert naive == after


def test_e2_t07_non_finite_covariates_never_reach_the_dataset() -> None:
    """An inf print is treated as missing; `pct` after a zero print is a DataError."""
    values = _trend()
    values[150] = np.inf
    cfg = _cfg([_covcfg("z", "Z")], standardize=False)
    covs = [_cov("Z", values)]
    data = Preparer(cfg).fit(_obs(), covs, TRAIN_END).transform(_obs(), covs)
    z = np.asarray(data.cov_transition)[:, 0]
    assert np.all(np.isfinite(z))
    assert z[150] == values[149]  # carried forward like a NaN print
    zero = _trend()
    zero[150] = 0.0
    cfg = _cfg([_covcfg("z", "Z", transform=["pct"])])
    with pytest.raises(DataError, match="pct") as info:
        Preparer(cfg).fit(_obs(), [_cov("Z", zero)], TRAIN_END)
    assert info.value.path == "data.covariates[0].transform"


def test_e2_t07_unsorted_and_revised_prints() -> None:
    """Prints are sorted before transforms; a duplicated timestamp keeps the last (revised) print."""
    cfg = _cfg([_covcfg("z", "Z", transform=["diff"])], standardize=False)
    values = _trend()
    reference = (
        Preparer(cfg).fit(_obs(), [_cov("Z", values)], TRAIN_END).transform(_obs(), [_cov("Z", values)])
    )
    order = np.random.default_rng(5).permutation(N_DAYS)
    shuffled = [_cov("Z", values[order], TIMES[order])]
    data = Preparer(cfg).fit(_obs(), shuffled, TRAIN_END).transform(_obs(), shuffled)
    np.testing.assert_array_equal(np.asarray(data.cov_transition), np.asarray(reference.cov_transition))
    # A revision of print 100 appended at the end replaces the original before the diff.
    revised = [_cov("Z", np.append(values, 99.0), np.append(TIMES, TIMES[100]))]
    data = Preparer(cfg).fit(_obs(), revised, TRAIN_END).transform(_obs(), revised)
    z = np.asarray(data.cov_transition)[:, 0]
    row = int(np.flatnonzero(data.times == TIMES[101])[0])  # row holding the print dated τ_100
    assert z[row] == 99.0 - values[99]
    assert z[row + 1] == values[101] - 99.0
