"""E2-T08: `regimekit fetch`, the data-quality report and `write_prepared`/`read_prepared`."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from click.testing import Result
from ruamel.yaml import YAML
from typer.testing import CliRunner

from regimekit.cli.app import app
from regimekit.config.yaml_io import load_config
from regimekit.data.contracts import BoundaryConstraints, CovariateSeries, ObservationSet, PreparedDataset
from regimekit.data.prepare import Preparer, prep_stats_from_json
from regimekit.finance import FinanceConfig
from regimekit.finance.quality import build_quality_report
from regimekit.persistence.results import read_prepared, write_prepared

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_YAML = YAML(typ="safe")
_runner = CliRunner()


def _raw_obs(path: Path) -> ObservationSet:
    """Read a price CSV verbatim (duplicates kept), as a provider that does not clean would return it."""
    frame = pd.read_csv(path)
    return ObservationSet(
        series_names=(path.stem,),
        times=pd.to_datetime(frame["time"]).to_numpy(dtype="datetime64[ns]"),
        values=frame[["close"]].to_numpy(dtype=np.float64),
        source="csv",
        snapshot_id="",
        meta={"interval": "1d", "adjusted": False},
    )


# --- AC1: every injected defect is reported with its timestamp ---------------------------------


def test_e2_t08_report_lists_each_injected_defect() -> None:
    """AC1: gap, duplicate, zero price and 10-sigma return are each listed with timestamps."""
    report = build_quality_report(_raw_obs(FIXTURES / "csv" / "defects" / "DEFECTS.csv"))
    assert report.gaps == ["2023-01-17", "2023-01-18", "2023-01-19"]
    assert report.duplicates == ["2023-02-01"]
    assert report.non_positive_prices == [{"time": "2023-02-10", "symbol": "DEFECTS", "price": 0.0}]
    assert [o["time"] for o in report.outliers] == ["2023-03-01"]
    assert report.outliers[0]["symbol"] == "DEFECTS"
    assert report.outliers[0]["sigmas"] > 8.0
    assert report.adjustment is None
    text = report.to_markdown()
    for stamp in ("2023-01-17", "2023-02-01", "2023-02-10", "2023-03-01"):
        assert stamp in text
    assert json.loads(report.to_json())["duplicates"] == ["2023-02-01"]


def test_e2_t08_clean_series_has_no_findings() -> None:
    """A clean weekday series produces empty defect lists."""
    dates = pd.bdate_range("2023-01-02", periods=80)
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 80)))
    obs = ObservationSet(
        series_names=("C",),
        times=dates.to_numpy(dtype="datetime64[ns]"),
        values=prices[:, None],
        source="csv",
        snapshot_id="",
        meta={"interval": "1d", "adjusted": True},
    )
    report = build_quality_report(obs)
    assert report.gaps == []
    assert report.duplicates == []
    assert report.non_positive_prices == []
    assert report.outliers == []


def test_e2_t08_adjustment_discontinuity_between_generations() -> None:
    """Overlapping prices of the two latest generations differing by > 1e-6 (relative) are flagged."""
    obs = _raw_obs(FIXTURES / "csv" / "defects" / "DEFECTS.csv")
    obs = dataclasses.replace(obs, snapshot_id="new")
    same = build_quality_report(obs, generations=(obs, dataclasses.replace(obs, snapshot_id="old")))
    assert same.adjustment is not None
    assert same.adjustment["discontinuous"] is False
    assert same.adjustment["max_rel_diff"] == 0.0
    earlier = dataclasses.replace(obs, snapshot_id="old", times=obs.times[:30], values=obs.values[:30] * 1.02)
    moved = build_quality_report(obs, generations=(obs, earlier))
    assert moved.adjustment is not None
    assert moved.adjustment["discontinuous"] is True
    assert moved.adjustment["latest"] == "new"
    assert moved.adjustment["previous"] == "old"
    assert moved.adjustment["n_overlap"] == 29  # 30 raw rows include the duplicated 2023-02-01
    assert moved.adjustment["max_rel_diff"] == pytest.approx(0.02 / 1.02, rel=1e-9)
    assert moved.adjustment["first_time"] == "2023-01-02"


def test_e2_t08_staleness_histogram_and_dropped_rows() -> None:
    """Staleness is histogrammed per covariate (-1 counted as `no_value`); drop counts are copied."""
    obs = _raw_obs(FIXTURES / "csv" / "defects" / "DEFECTS.csv")
    report = build_quality_report(
        obs,
        staleness={"z": np.array([-1, 0, 1, 1, 2, 0])},
        dropped_rows={"missing_price": 1, "warmup": 0, "covariate:z": 3},
    )
    assert report.staleness == {"z": {"0": 2, "1": 2, "2": 1, "no_value": 1}}
    assert report.dropped_rows == {"missing_price": 1, "warmup": 0, "covariate:z": 3}
    assert "covariate:z" in report.to_markdown()


def test_e2_t08_report_written_as_json_and_markdown(tmp_path: Path) -> None:
    """`write` produces `quality.json` and `quality.md`."""
    report = build_quality_report(_raw_obs(FIXTURES / "csv" / "defects" / "DEFECTS.csv"))
    json_path, md_path = report.write(tmp_path)
    assert json_path == tmp_path / "quality.json"
    assert md_path == tmp_path / "quality.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["gaps"] == report.gaps
    assert md_path.read_text(encoding="utf-8").startswith("# Data-quality report")


# --- AC2 / AC3: the fetch command ----------------------------------------------------------------

N_DAYS = 300


def _write_csv_project(
    tmp_path: Path, *, missing: str = "ffill", gap: int = 0, **cfg_updates: object
) -> Path:
    """Synthetic csv project: `PRIMARY.csv`, `VIXLIKE.csv`, `SPREAD.csv` and an Appendix A-based config."""
    dates = pd.bdate_range("2020-01-02", periods=N_DAYS).strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, N_DAYS)))
    pd.DataFrame({"time": dates, "close": close, "adj_close": close}).to_csv(
        tmp_path / "PRIMARY.csv", index=False
    )
    vix = 20 + np.cumsum(rng.normal(0, 0.5, N_DAYS)).clip(-10, 30)
    vix_text = [f"{v:.4f}" for v in vix]
    vix_text[50] = "."  # FRED-style holiday
    for i in range(gap):
        vix_text[150 + i] = "."
    pd.DataFrame({"time": dates, "value": vix_text}).to_csv(tmp_path / "VIXLIKE.csv", index=False)
    spread = np.sin(np.arange(N_DAYS) / 15.0)
    pd.DataFrame({"time": dates, "value": spread}).to_csv(tmp_path / "SPREAD.csv", index=False)

    cfg = _YAML.load((FIXTURES / "configs" / "appendix_a.yaml").read_text(encoding="utf-8"))
    cfg["project"]["output_dir"] = str(tmp_path / "runs")
    data = cfg["data"]
    data["primary"].update(provider="csv", symbols=["PRIMARY"], start="2020-01-01", end=None)
    data["cache"]["dir"] = str(tmp_path / "cache")
    data["covariates"][0].update(provider="csv", series="VIXLIKE", missing=missing)
    data["covariates"][1].update(provider="csv", series="SPREAD")
    cfg["evaluation"]["backtest"]["initial_train_years"] = 0.5
    for dotted, value in cfg_updates.items():
        node = cfg
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as fh:
        _YAML.dump(cfg, fh)
    return path


def _fetch(config: Path, out: Path, *extra: str) -> Result:
    return _runner.invoke(app, ["--out", str(out), *extra, "fetch", str(config)])


def test_e2_t08_fetch_writes_all_outputs(tmp_path: Path) -> None:
    """AC2: `fetch` on a csv config writes every artifact and exits 0."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    result = _fetch(config, out)
    assert result.exit_code == 0, result.output
    for name in ("prepared.parquet", "prep_stats.json", "snapshots.json", "quality.json", "quality.md"):
        assert (out / name).is_file(), name

    ds = read_prepared(out / "prepared.parquet")
    ds.validate()
    assert ds.cov_names == {"transition": ("log_vix", "term_spread"), "duration": ("log_vix",)}
    assert prep_stats_from_json((out / "prep_stats.json").read_text(encoding="utf-8")) == ds.prep_stats
    # Standardization window = the first 0.5 years of the primary.
    assert ds.prep_stats.train_window[1] <= "2020-07-02"

    snapshots = json.loads((out / "snapshots.json").read_text(encoding="utf-8"))
    ids = [snapshots["primary"]["snapshot_id"], *(c["snapshot_id"] for c in snapshots["covariates"])]
    assert tuple(ids) == ds.snapshot_ids
    assert all(len(i) == 64 for i in ids)
    assert [c["name"] for c in snapshots["covariates"]] == ["log_vix", "term_spread"]

    quality = json.loads((out / "quality.json").read_text(encoding="utf-8"))
    assert set(quality["staleness"]) == {"log_vix", "term_spread"}
    assert quality["staleness"]["log_vix"]["1"] >= 1  # the FRED-style holiday was carried forward
    assert quality["dropped_rows"]["missing_price"] == 0
    assert "Local files supplied by the user" in result.output  # ToS note logged


def test_e2_t08_offline_refetch_is_bitwise_identical(tmp_path: Path) -> None:
    """A second `--offline` fetch from the cache reproduces `prepared.parquet` byte for byte."""
    config = _write_csv_project(tmp_path)
    assert _fetch(config, tmp_path / "a").exit_code == 0
    (tmp_path / "PRIMARY.csv").unlink()  # offline must not touch the provider
    result = _fetch(config, tmp_path / "b", "--offline")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "a" / "prepared.parquet").read_bytes() == (
        tmp_path / "b" / "prepared.parquet"
    ).read_bytes()


def test_e2_t08_missing_error_with_stale_covariate_exits_3(tmp_path: Path) -> None:
    """AC3: `missing: error` with a covariate stale beyond `max_staleness` exits 3 (report still written)."""
    config = _write_csv_project(tmp_path, missing="error", gap=10)
    out = tmp_path / "out"
    result = _fetch(config, out)
    assert result.exit_code == 3, result.output
    assert "missing='error'" in result.output
    assert (out / "quality.json").is_file()
    assert not (out / "prepared.parquet").exists()
    dropped = json.loads((out / "quality.json").read_text(encoding="utf-8"))["dropped_rows"]
    assert dropped == {"missing_price": 0}  # the policies did not complete: nothing else is claimed


def test_e2_t08_failed_run_removes_outputs_of_an_earlier_run(tmp_path: Path) -> None:
    """A failing fetch into a directory used by a successful one leaves no stale dataset behind."""
    out = tmp_path / "out"
    assert _fetch(_write_csv_project(tmp_path), out).exit_code == 0
    assert (out / "prepared.parquet").is_file()
    # max_age_days 0: refetch, so the cache serves the new (gapped) covariate file.
    result = _fetch(_write_csv_project(tmp_path, missing="error", gap=10, data__cache__max_age_days=0), out)
    assert result.exit_code == 3, result.output
    for name in ("prepared.parquet", "prep_stats.json", "snapshots.json"):
        assert not (out / name).exists(), name
    assert (out / "quality.json").is_file()


def test_e2_t08_unwritable_output_is_a_persistence_error(tmp_path: Path) -> None:
    """An output path that is a file maps to PersistenceError (exit 1), not a traceback."""
    blocker = tmp_path / "out"
    blocker.write_text("not a directory", encoding="utf-8")
    result = _fetch(_write_csv_project(tmp_path), blocker)
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "cannot" in result.output


def test_e2_t08_empty_primary_request_is_a_data_error(tmp_path: Path) -> None:
    """A primary window with fewer than two rows exits 3."""
    config = _write_csv_project(tmp_path, data__primary__start="2025-01-01")
    result = _fetch(config, tmp_path / "out")
    assert result.exit_code == 3, result.output


def test_e2_t08_provider_error_exits_5(tmp_path: Path) -> None:
    """A provider failure (missing CSV file) exits 5."""
    config = _write_csv_project(tmp_path)
    (tmp_path / "SPREAD.csv").unlink()
    result = _fetch(config, tmp_path / "out")
    assert result.exit_code == 5, result.output
    assert "SPREAD" in result.output


def test_e2_t08_pinned_snapshot_with_covariates_is_rejected(tmp_path: Path) -> None:
    """A pinned `data.cache.snapshot` cannot cover covariate entries yet (owner decision): exit 2."""
    config = _write_csv_project(tmp_path, data__cache__snapshot="a" * 64)
    result = _fetch(config, tmp_path / "out")
    assert result.exit_code == 2, result.output
    assert "data.cache.snapshot" in result.output


def test_e2_t08_pinned_snapshot_replays_primary_only_run(tmp_path: Path) -> None:
    """Without covariates a pinned snapshot id replays the run from the cache."""
    config = _write_csv_project(
        tmp_path, data__covariates=[], model__transition__covariates=[], model__duration__covariates=[]
    )
    cfg = load_config(config, FinanceConfig)
    assert cfg.data.covariates == []
    assert _fetch(config, tmp_path / "a").exit_code == 0
    sid = json.loads((tmp_path / "a" / "snapshots.json").read_text(encoding="utf-8"))["primary"][
        "snapshot_id"
    ]
    pinned = _write_csv_project(
        tmp_path,
        data__covariates=[],
        model__transition__covariates=[],
        model__duration__covariates=[],
        data__cache__snapshot=sid,
    )
    (tmp_path / "PRIMARY.csv").unlink()
    result = _fetch(pinned, tmp_path / "b")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "a" / "prepared.parquet").read_bytes() == (
        tmp_path / "b" / "prepared.parquet"
    ).read_bytes()


# --- AC4: prepared dataset round trip ------------------------------------------------------------


def _prepared(clock: str = "trading") -> PreparedDataset:
    cfg = load_config(FIXTURES / "configs" / "appendix_a.yaml", FinanceConfig).data
    options = cfg.clock_options if clock == "trading" else {}  # only the trading clock has options
    cfg = cfg.model_copy(update={"clock": clock, "clock_options": options})
    dates = pd.bdate_range("2021-01-04", periods=120)
    times = dates.to_numpy(dtype="datetime64[ns]")
    rng = np.random.default_rng(3)
    obs = ObservationSet(
        series_names=("X",),
        times=times,
        values=(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))[:, None],
        source="csv",
        snapshot_id="p" * 64,
        meta={"interval": "1d", "adjusted": True},
    )
    covs = [
        CovariateSeries("VIXCLS", times, 20 + rng.random(120), "csv", "VIXCLS", "v" * 64, "d"),
        CovariateSeries("T10Y2Y", times, rng.normal(size=120), "csv", "T10Y2Y", "t" * 64, "d"),
    ]
    return Preparer(cfg).fit(obs, covs, "2021-04-01").transform(obs, covs)


def _assert_bitwise(a: PreparedDataset, b: PreparedDataset) -> None:
    for name in (
        "times",
        "observations",
        "dt",
        "mask",
        "cov_transition",
        "cov_duration",
        "cov_emission",
        "labels",
    ):
        x, y = getattr(a, name), getattr(b, name)
        if x is None:
            assert y is None, name
            continue
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype, name
        assert x.shape == y.shape, name
        assert x.tobytes() == y.tobytes(), name


@pytest.mark.parametrize("clock", ["trading", "calendar", "steps"])
def test_e2_t08_prepared_round_trip_is_bitwise(tmp_path: Path, clock: str) -> None:
    """AC4: `read_prepared(write_prepared(ds)) == ds` with every array bitwise identical."""
    ds = _prepared(clock)
    back = read_prepared(write_prepared(ds, tmp_path / "prepared.parquet"))
    assert back == ds
    _assert_bitwise(ds, back)
    assert back.nominal_dt == ds.nominal_dt
    back.validate()


def test_e2_t08_prepared_round_trip_part_ii_fields(tmp_path: Path) -> None:
    """The Part II hooks (`labels`, `boundaries`, `group`) and an emission block survive the round trip."""
    ds = _prepared()
    t = ds.times.shape[0]
    ds = dataclasses.replace(
        ds,
        cov_emission=jnp.asarray(np.asarray(ds.cov_transition)[:, :1]),
        cov_names={**ds.cov_names, "emission": ("log_vix",)},
        labels=jnp.asarray(np.random.default_rng(0).random((t, 3)) > 0.5),
        boundaries=BoundaryConstraints(must_switch_at=(3, 9), must_not_switch_at=(5,)),
        group="dax",
    )
    ds.validate()
    back = read_prepared(write_prepared(ds, tmp_path / "prepared.parquet"))
    assert back == ds
    _assert_bitwise(ds, back)


def test_e2_t08_refetch_reports_adjustment_discontinuity(tmp_path: Path) -> None:
    """A refetch whose history moved (a new dividend adjustment) is flagged against the old generation."""
    config = _write_csv_project(tmp_path, data__cache__max_age_days=0)
    assert _fetch(config, tmp_path / "a").exit_code == 0
    first = json.loads((tmp_path / "a" / "quality.json").read_text(encoding="utf-8"))
    assert first["adjustment"] is None
    frame = pd.read_csv(tmp_path / "PRIMARY.csv")
    frame["adj_close"] = frame["adj_close"] * 0.99
    frame.to_csv(tmp_path / "PRIMARY.csv", index=False)
    assert _fetch(config, tmp_path / "b").exit_code == 0
    adjustment = json.loads((tmp_path / "b" / "quality.json").read_text(encoding="utf-8"))["adjustment"]
    snapshots = json.loads((tmp_path / "b" / "snapshots.json").read_text(encoding="utf-8"))
    assert adjustment["latest"] == snapshots["primary"]["snapshot_id"]
    assert adjustment["discontinuous"] is True
    assert adjustment["n_overlap"] == N_DAYS
    assert adjustment["max_rel_diff"] == pytest.approx(0.01, rel=1e-9)
    assert adjustment["first_time"] == "2020-01-02"
    assert "DISCONTINUOUS" in (tmp_path / "b" / "quality.md").read_text(encoding="utf-8")


def test_e2_t08_quota_providers_keep_quota_in_cache_dir(tmp_path: Path) -> None:
    """stooq/tiingo are built with `quota_dir` = the cache directory (no network at construction)."""
    from regimekit.finance.providers.source import make_provider as _make_provider

    data = load_config(_write_csv_project(tmp_path), FinanceConfig).data
    cache_dir = Path(data.cache.dir).expanduser()
    stooq = _make_provider("stooq", data, tmp_path)
    assert stooq._quota_file == cache_dir / "stooq_quota.json"  # type: ignore[attr-defined]
    tiingo = _make_provider("tiingo", data, tmp_path)
    assert tiingo.quota.path == cache_dir / "tiingo_quota.json"  # type: ignore[attr-defined]


def test_e2_t08_report_edge_cases() -> None:
    """Weekly data has no weekday-gap check; too-short or constant series flag no outliers."""
    dates = pd.date_range("2023-01-06", periods=10, freq="W-FRI").to_numpy(dtype="datetime64[ns]")
    weekly = ObservationSet(
        ("W",), dates, np.linspace(1, 2, 10)[:, None], "csv", "", {"interval": "1wk", "adjusted": True}
    )
    assert build_quality_report(weekly).gaps == []
    short = dataclasses.replace(
        weekly,
        times=dates[:3],
        values=np.array([[1.0], [2.0], [3.0]]),
        meta={"interval": "1d", "adjusted": True},
    )
    assert build_quality_report(short).outliers == []
    flat = dataclasses.replace(weekly, values=np.ones((10, 1)))
    assert build_quality_report(flat).outliers == []


def test_e2_t08_read_prepared_errors(tmp_path: Path) -> None:
    """Missing files, foreign parquet files and unknown format versions raise PersistenceError."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from regimekit.utils.errors import PersistenceError

    with pytest.raises(PersistenceError, match="cannot read"):
        read_prepared(tmp_path / "missing.parquet")
    pq.write_table(pa.table({"x": [1.0]}), tmp_path / "foreign.parquet")
    with pytest.raises(PersistenceError, match="not a regimekit"):
        read_prepared(tmp_path / "foreign.parquet")
    path = write_prepared(_prepared(), tmp_path / "prepared.parquet")
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"regimekit"])
    meta["format"] = 99
    pq.write_table(table.replace_schema_metadata({b"regimekit": json.dumps(meta).encode()}), path)
    with pytest.raises(PersistenceError, match="format"):
        read_prepared(path)


def test_e2_t08_prepared_round_trip_empty_blocks(tmp_path: Path) -> None:
    """`(T, 0)` covariate and label blocks (CLAUDE.md §3.7) round-trip with their shape and dtype."""
    ds = _prepared()
    t = ds.times.shape[0]
    ds = dataclasses.replace(
        ds,
        cov_emission=jnp.zeros((t, 0)),
        cov_names={**ds.cov_names, "emission": ()},
        labels=jnp.zeros((t, 0), dtype=bool),
    )
    ds.validate()
    back = read_prepared(write_prepared(ds, tmp_path / "prepared.parquet"))
    assert back == ds
    _assert_bitwise(ds, back)
