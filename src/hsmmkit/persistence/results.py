"""Result writers/readers: the prepared dataset of `hsmmkit fetch`, decode, forecast and simulation outputs.

`prepared.parquet` layout: one row per observation; columns `time` (timestamp[ns]), `obs_<m>`,
`mask_<m>`, `dt`, `cov_<entry point>_<k>` and `label_<n>`; everything else (`clock`, `nominal_dt`,
covariate names per entry point, `snapshot_ids`, `prep_stats`, the Part II hooks) is JSON in the
schema metadata under `hsmmkit`. Arrays round-trip bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hsmmkit.data.contracts import BoundaryConstraints, PreparedDataset
from hsmmkit.data.prepare import prep_stats_from_json, prep_stats_to_json
from hsmmkit.evaluation.results import BacktestResult
from hsmmkit.inference.results import ForecastResult
from hsmmkit.utils.errors import PersistenceError

if TYPE_CHECKING:
    from hsmmkit.inference.results import PosteriorResult

_META_KEY = b"hsmmkit"
_FORMAT = 1
_ENTRY_POINTS = ("transition", "duration", "emission")


def write_prepared(dataset: PreparedDataset, path: Path | str) -> Path:
    """Write `dataset` to a parquet file (parent directories are created).

    Returns:
        The written path (so `read_prepared(write_prepared(ds, p)) == ds`).
    """
    path = Path(path)
    observations = np.asarray(dataset.observations)
    mask = np.asarray(dataset.mask)
    columns: dict[str, Any] = {"time": pa.array(np.asarray(dataset.times, dtype="datetime64[ns]"))}
    for m in range(observations.shape[1]):
        columns[f"obs_{m}"] = pa.array(observations[:, m])
        columns[f"mask_{m}"] = pa.array(mask[:, m])
    columns["dt"] = pa.array(np.asarray(dataset.dt))
    blocks: dict[str, int | None] = {}
    for ep in _ENTRY_POINTS:
        block = getattr(dataset, f"cov_{ep}")
        blocks[ep] = None if block is None else int(block.shape[1])
        if block is not None:
            values = np.asarray(block)
            for k in range(values.shape[1]):
                columns[f"cov_{ep}_{k}"] = pa.array(values[:, k])
    n_labels = None
    if dataset.labels is not None:
        labels = np.asarray(dataset.labels)
        n_labels = int(labels.shape[1])
        for n in range(n_labels):
            columns[f"label_{n}"] = pa.array(labels[:, n])
    boundaries = dataset.boundaries
    meta = {
        "format": _FORMAT,
        "n_series": int(observations.shape[1]),
        "clock": dataset.clock,
        "nominal_dt": dataset.nominal_dt,
        "blocks": blocks,
        "cov_names": {k: list(v) for k, v in dataset.cov_names.items()},
        "snapshot_ids": list(dataset.snapshot_ids),
        "prep_stats": prep_stats_to_json(dataset.prep_stats),
        "support": dataset.support,
        "n_categories": dataset.n_categories,
        "n_labels": n_labels,
        "boundaries": None
        if boundaries is None
        else {
            "must_switch_at": list(boundaries.must_switch_at),
            "must_not_switch_at": list(boundaries.must_not_switch_at),
        },
        "group": dataset.group,
    }
    table = pa.table(columns).replace_schema_metadata({_META_KEY: json.dumps(meta).encode("utf-8")})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_prepared(path: Path | str) -> PreparedDataset:
    """Read a dataset written by `write_prepared`.

    Raises:
        PersistenceError: missing file, not a hsmmkit dataset, or an unsupported format version.
    """
    path = Path(path)
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowInvalid) as exc:
        raise PersistenceError(f"cannot read prepared dataset {path}: {exc}", path=str(path)) from exc
    raw = (table.schema.metadata or {}).get(_META_KEY)
    if raw is None:
        raise PersistenceError(f"{path} is not a hsmmkit prepared dataset", path=str(path))
    meta = json.loads(raw)
    if meta.get("format") != _FORMAT:
        raise PersistenceError(f"unsupported prepared-dataset format {meta.get('format')!r}", path=str(path))

    def col(name: str) -> np.ndarray:
        return np.asarray(table.column(name).to_numpy())

    def stack(names: list[str], dtype: type) -> np.ndarray:
        if not names:  # a (T, 0) block (CLAUDE.md §3.7) has no columns
            return np.empty((table.num_rows, 0), dtype=dtype)
        return np.stack([col(n) for n in names], axis=1)

    n_series = meta["n_series"]
    observations = np.stack([col(f"obs_{m}") for m in range(n_series)], axis=1)
    mask = np.stack([col(f"mask_{m}") for m in range(n_series)], axis=1)
    blocks: dict[str, Any] = {}
    for ep in _ENTRY_POINTS:
        k = meta["blocks"][ep]
        blocks[ep] = (
            None if k is None else jnp.asarray(stack([f"cov_{ep}_{i}" for i in range(k)], np.float64))
        )
    labels = None
    if meta["n_labels"] is not None:
        labels = jnp.asarray(stack([f"label_{n}" for n in range(meta["n_labels"])], np.bool_))
    b = meta["boundaries"]
    return PreparedDataset(
        times=col("time").astype("datetime64[ns]"),
        observations=jnp.asarray(observations),
        dt=jnp.asarray(col("dt")),
        mask=jnp.asarray(mask),
        cov_transition=blocks["transition"],
        cov_duration=blocks["duration"],
        cov_emission=blocks["emission"],
        cov_names={k: tuple(v) for k, v in meta["cov_names"].items()},
        clock=meta["clock"],
        nominal_dt=meta["nominal_dt"],
        prep_stats=prep_stats_from_json(meta["prep_stats"]),
        snapshot_ids=tuple(meta["snapshot_ids"]),
        support=meta["support"],
        n_categories=meta["n_categories"],
        labels=labels,
        boundaries=None
        if b is None
        else BoundaryConstraints(tuple(b["must_switch_at"]), tuple(b["must_not_switch_at"])),
        group=meta["group"],
    )


def write_posterior(result: PosteriorResult, out_dir: Path | str, *, augmented: bool = False) -> None:
    """Write `decode`'s `PosteriorResult` (E6-T02): `posterior.parquet`, `segments.json`, `criteria.json`.

    `posterior.parquet` has `T` rows and `2N+2` columns: `time`, `filtered_0..N-1`, `smoothed_0..N-1`,
    `viterbi` (the regime path). `augmented=True` also writes `augmented.parquet` (`time` x `S` smoothed
    expanded-state posterior, `state_0..S-1`).

    Raises:
        PersistenceError: the output directory cannot be written.
    """
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        filtered = np.asarray(result.filtered)
        smoothed = np.asarray(result.smoothed)
        n = filtered.shape[1]
        time_col = pa.array(np.asarray(result.times, dtype="datetime64[ns]"))
        columns: dict[str, Any] = {"time": time_col}
        for i in range(n):
            columns[f"filtered_{i}"] = pa.array(filtered[:, i])
        for i in range(n):
            columns[f"smoothed_{i}"] = pa.array(smoothed[:, i])
        columns["viterbi"] = pa.array(np.asarray(result.viterbi))
        pq.write_table(pa.table(columns), out / "posterior.parquet")

        if augmented:
            aug_smoothed = np.asarray(result.augmented.smoothed)
            aug_columns: dict[str, Any] = {"time": time_col}
            for j in range(aug_smoothed.shape[1]):
                aug_columns[f"state_{j}"] = pa.array(aug_smoothed[:, j])
            pq.write_table(pa.table(aug_columns), out / "augmented.parquet")

        segments_payload = [{"start": s, "end": e, "regime": r} for s, e, r in result.segments]
        (out / "segments.json").write_text(json.dumps(segments_payload, indent=2), encoding="utf-8")
        criteria_payload = {
            "loglik": result.loglik,
            "aic": result.aic,
            "bic": result.bic,
            "icl": result.icl,
        }
        (out / "criteria.json").write_text(json.dumps(criteria_payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot write decode outputs to {out}: {exc}", path=str(out)) from exc


_FORECAST_COLUMNS = ("origin_time", "horizon_time", "step", "kind", "key", "series", "value")


def write_forecast(
    result: ForecastResult,
    out_dir: Path | str,
    *,
    samples: bool = False,
    snapshot_ids: tuple[str, ...] = (),
) -> None:
    """Write `forecast.parquet` (long format), `forecast.json` and optionally `forecast_samples.parquet`.

    `forecast.parquet` columns: `origin_time`, `horizon_time` (timestamp[ns]), `step` (int64, 1-based),
    `kind` (`regime_prob` | `quantile` | `mean` | `var`), `key` (string: regime index, quantile level as
    `repr(float)`, or series index), `series` (int64; 0 for `regime_prob`) and `value` (float64, bitwise).
    `forecast_samples.parquet` (`samples=True`): `path`, `step`, `series`, `value`.

    Raises:
        PersistenceError: the output directory cannot be written.
    """
    out = Path(out_dir)
    probs = np.asarray(result.regime_probs)
    quant = np.asarray(result.cum_quantiles)
    mean, var = np.asarray(result.mean), np.asarray(result.var)
    n_sim, h, m = np.asarray(result.samples).shape
    rows: list[tuple[int, str, str, int, float]] = []
    for k in range(h):
        rows += [(k, "regime_prob", str(i), 0, float(probs[k, i])) for i in range(probs.shape[1])]
        for qi, level in enumerate(result.quantile_levels):
            rows += [(k, "quantile", repr(float(level)), s, float(quant[k, qi, s])) for s in range(m)]
        rows += [(k, "mean", str(s), s, float(mean[k, s])) for s in range(m)]
        rows += [(k, "var", str(s), s, float(var[k, s])) for s in range(m)]
    origin = np.datetime64(result.origin_time, "ns")
    horizon = np.asarray(result.horizon_times, dtype="datetime64[ns]")
    try:
        out.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "origin_time": pa.array(np.full(len(rows), origin, dtype="datetime64[ns]")),
                "horizon_time": pa.array(horizon[[r[0] for r in rows]]),
                "step": pa.array(np.asarray([r[0] + 1 for r in rows], dtype=np.int64)),
                "kind": pa.array([r[1] for r in rows], type=pa.string()),
                "key": pa.array([r[2] for r in rows], type=pa.string()),
                "series": pa.array(np.asarray([r[3] for r in rows], dtype=np.int64)),
                "value": pa.array(np.asarray([r[4] for r in rows], dtype=np.float64)),
            }
        )
        pq.write_table(table, out / "forecast.parquet")
        summary = {
            "origin_time": str(origin),
            "horizon": h,
            "n_sim": n_sim,
            "n_series": m,
            "n_states": int(probs.shape[1]),
            "covariate_forecaster": result.covariate_forecaster,
            "seed": result.seed,
            "quantile_levels": [float(q) for q in result.quantile_levels],
            "snapshot_ids": list(snapshot_ids),
        }
        (out / "forecast.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        samples_path = out / "forecast_samples.parquet"
        if samples:
            arr = np.asarray(result.samples, dtype=np.float64)
            path_idx, step_idx, series_idx = np.meshgrid(
                np.arange(n_sim), np.arange(h), np.arange(m), indexing="ij"
            )
            pq.write_table(
                pa.table(
                    {
                        "path": pa.array(path_idx.reshape(-1).astype(np.int64)),
                        "step": pa.array(step_idx.reshape(-1).astype(np.int64) + 1),
                        "series": pa.array(series_idx.reshape(-1).astype(np.int64)),
                        "value": pa.array(arr.reshape(-1)),
                    }
                ),
                samples_path,
            )
        elif samples_path.exists():
            samples_path.unlink()  # a stale file from an earlier run must not pair with this forecast
    except OSError as exc:
        raise PersistenceError(f"cannot write forecast outputs to {out}: {exc}", path=str(out)) from exc


def read_forecast(path: Path | str) -> ForecastResult:
    """Read a `write_forecast` directory back into a `ForecastResult` (values bitwise).

    `samples` is restored from `forecast_samples.parquet` when present, else it is an empty
    `(0, h, M)` array (`forecast.json` keeps the original `n_sim`).

    Raises:
        PersistenceError: the directory holds no readable forecast.
    """
    src = Path(path)
    try:
        table = pq.read_table(src / "forecast.parquet").to_pandas()
        meta = json.loads((src / "forecast.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise PersistenceError(f"{src} holds no readable forecast: {exc}", path=str(src)) from exc

    try:
        h, m, n = int(meta["horizon"]), int(meta["n_series"]), int(meta["n_states"])
        levels = tuple(float(q) for q in meta["quantile_levels"])

        step = table["step"].to_numpy(dtype=np.int64) - 1
        kind = table["kind"].to_numpy(dtype=object)
        key = table["key"].to_numpy(dtype=object)
        series = table["series"].to_numpy(dtype=np.int64)
        value = table["value"].to_numpy(dtype=np.float64)
        level_pos = {repr(q): i for i, q in enumerate(levels)}

        probs = np.zeros((h, n))
        quant = np.zeros((h, len(levels), m))
        mean, var = np.zeros((h, m)), np.zeros((h, m))
        for i in range(len(table)):
            if kind[i] == "regime_prob":
                probs[step[i], int(key[i])] = value[i]
            elif kind[i] == "quantile":
                quant[step[i], level_pos[key[i]], series[i]] = value[i]
            elif kind[i] == "mean":
                mean[step[i], series[i]] = value[i]
            elif kind[i] == "var":
                var[step[i], series[i]] = value[i]
        horizon_times = (
            table.drop_duplicates("step")
            .sort_values("step")["horizon_time"]
            .to_numpy()
            .astype("datetime64[ns]")
        )

        samples_path = src / "forecast_samples.parquet"
        if samples_path.exists():
            flat = pq.read_table(samples_path).to_pandas().sort_values(["path", "step", "series"])
            samples = flat["value"].to_numpy().reshape(-1, h, m)
        else:
            samples = np.zeros((0, h, m))

    except (KeyError, IndexError, ValueError) as exc:
        raise PersistenceError(f"{src} holds a malformed forecast: {exc!r}", path=str(src)) from exc

    return ForecastResult(
        origin_time=np.datetime64(meta["origin_time"], "ns"),
        horizon_times=horizon_times,
        regime_probs=jnp.asarray(probs),
        samples=jnp.asarray(samples),
        cum_quantiles=jnp.asarray(quant),
        mean=jnp.asarray(mean),
        var=jnp.asarray(var),
        quantile_levels=levels,
        covariate_forecaster=str(meta["covariate_forecaster"]),
        seed=int(meta["seed"]),
    )


_VALUE_COLUMN = "return"  # column of `simulation.parquet` (Part I name, kept for compatibility)


def write_simulation(
    out_dir: Path | str,
    *,
    times: np.ndarray,
    regimes: np.ndarray,
    phases: np.ndarray,
    values: np.ndarray,
    segments: list[list[tuple[int, int, int]]],
) -> None:
    """Write `simulation.parquet` and `simulation_segments.json`.

    `simulation.parquet` is long: `path`, `step`, `time`, `regime`, `phase`, `return`
    (the Part I column name, kept for compatibility). `regimes`/`phases` are `(P, T)`, `values`
    `(P, T, M)`, `times` `(T,)`; `step` is 1-based. With `M > 1` the column is split into one per series
    (`<column>_0 .. <column>_{M-1}`). `phase` is the 1-based phase within the regime
    (`StateIndex.phase_of`); segments are `{start, end, regime}`, `end` inclusive.

    Raises:
        PersistenceError: the output directory cannot be written.
    """
    out = Path(out_dir)
    n_paths, t_len = regimes.shape
    m = values.shape[2]
    columns: dict[str, Any] = {
        "path": pa.array(np.repeat(np.arange(n_paths, dtype=np.int64), t_len)),
        "step": pa.array(np.tile(np.arange(1, t_len + 1, dtype=np.int64), n_paths)),
        "time": pa.array(np.tile(np.asarray(times, dtype="datetime64[ns]"), n_paths)),
        "regime": pa.array(regimes.reshape(-1).astype(np.int64)),
        "phase": pa.array(phases.reshape(-1).astype(np.int64)),
    }
    if m == 1:
        columns[_VALUE_COLUMN] = pa.array(values[:, :, 0].reshape(-1).astype(np.float64))
    else:
        for s in range(m):
            columns[f"{_VALUE_COLUMN}_{s}"] = pa.array(values[:, :, s].reshape(-1).astype(np.float64))
    payload = [
        {"path": p, "segments": [{"start": a, "end": b, "regime": r} for a, b, r in segs]}
        for p, segs in enumerate(segments)
    ]
    try:
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out / "simulation.parquet")
        (out / "simulation_segments.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot write simulation outputs to {out}: {exc}", path=str(out)) from exc


# --- backtest (E7-T04) -------------------------------------------------------------------------------

_BACKTEST_PARQUET = {
    "metrics": "backtest_metrics.parquet",
    "folds": "backtest_folds.parquet",
    "summary": "backtest_summary.parquet",
}
BACKTEST_META = "backtest.json"


def _table(frame: pd.DataFrame) -> pa.Table:
    """Arrow table of `frame` (no index): `timestamp[ns]` for datetimes, plain `string` for text."""
    cast = {
        c: frame[c].astype("datetime64[ns]")
        for c in frame.columns
        if str(frame[c].dtype).startswith("datetime64")
    }
    table = pa.Table.from_pandas(frame.assign(**cast), preserve_index=False)
    fields = [pa.field(f.name, pa.string()) if pa.types.is_large_string(f.type) else f for f in table.schema]
    return table.cast(pa.schema(fields))


def _stale_extras(out: Path) -> list[str]:
    """Extras frames of an earlier run in `out` (listed in its `backtest.json`)."""
    try:
        previous = json.loads((out / BACKTEST_META).read_text(encoding="utf-8"))
        return [str(n) for n in previous.get("extras", [])]
    except (OSError, ValueError, AttributeError):
        return []


def write_backtest(result: BacktestResult, out_dir: Path | str, meta: dict[str, Any]) -> None:
    """Write a `BacktestResult` and its metadata.

    Files: `backtest_metrics.parquet` (long: `fold, horizon, metric, value`), `backtest_folds.parquet`,
    `backtest_summary.parquet` (`horizon, metric, mean, sd, n`), one `<name>.parquet` per entry of
    `result.extras` (the names are listed under `extras` in `backtest.json`; files of an earlier run that
    are not in `result.extras` are removed) and `backtest.json` (`meta`).

    `horizon == 0` marks per-fold model and hook metrics. `meta` must be JSON-serializable.

    Raises:
        PersistenceError: the output directory cannot be written.
    """
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(_table(result.metrics), out / _BACKTEST_PARQUET["metrics"])
        pq.write_table(_table(result.folds), out / _BACKTEST_PARQUET["folds"])
        pq.write_table(_table(result.summary), out / _BACKTEST_PARQUET["summary"])
        for old in _stale_extras(out):
            (out / f"{old}.parquet").unlink(missing_ok=True)
        for name, frame in result.extras.items():
            pq.write_table(_table(frame), out / f"{name}.parquet")
        stored = {**meta, "extras": sorted(result.extras)} if result.extras else dict(meta)
        (out / BACKTEST_META).write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot write backtest outputs to {out}: {exc}", path=str(out)) from exc


def read_backtest(out_dir: Path | str) -> tuple[BacktestResult, dict[str, Any]]:
    """Restore `(BacktestResult, meta)` written by `write_backtest`.

    Raises:
        PersistenceError: a file is missing or unreadable.
    """
    out = Path(out_dir)
    try:
        frames = {k: pq.read_table(out / name).to_pandas() for k, name in _BACKTEST_PARQUET.items()}
        meta = json.loads((out / BACKTEST_META).read_text(encoding="utf-8"))
        names = list(meta.get("extras", [])) if isinstance(meta, dict) else []
        extras = {name: pq.read_table(out / f"{name}.parquet").to_pandas() for name in names}
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise PersistenceError(f"cannot read backtest outputs from {out}: {exc}", path=str(out)) from exc
    if not isinstance(meta, dict) or "snapshot_ids" not in meta:
        raise PersistenceError(f"{out / BACKTEST_META} has no snapshot_ids", path=str(out))
    meta.pop("extras", None)
    return BacktestResult(frames["folds"], frames["metrics"], frames["summary"], extras), meta
