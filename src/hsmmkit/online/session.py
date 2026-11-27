"""`OnlineSession`: rows in, regime probabilities and alert events out (E15-T04).

Concept refs: Part II §14.11. The session turns raw rows (a mapping / `pandas.Series` with the observation
columns, an optional time column and the covariate columns of the configuration) into the arguments of
`ExpandedOnlineFilter.update_from_time`: observations (standardized with the fitted `prep_stats` if the model
was fitted on standardized observations), the lagged covariate rows (`OnlineCovariatePreparer`) and the
clock's `dt`. It backs `Project.online()` and the `stream` command.

Supported: the identity observation transform and no own-lag emissions (otherwise `ConfigError`). Covariates
are dated by row (`t + 1 - lag`, as a time-less offline alignment); leading rows for which a covariate has no
print yet are skipped (the offline `Preparer` trims them too) when the session starts cold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hsmmkit.data.sources.base import parse_times
from hsmmkit.online.covariates import OnlineCovariatePreparer
from hsmmkit.online.filter import ExpandedOnlineFilter
from hsmmkit.online.state import load_state, save_state
from hsmmkit.utils.clock import get_clock
from hsmmkit.utils.errors import ConfigError, DataError

if TYPE_CHECKING:
    from hsmmkit.config.schema import ProjectConfig
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.inference.results import ForecastResult
    from hsmmkit.online.alerts import AlertEvent

SESSION_FILE = "session.json"


def _time_json(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="ns"))
    if isinstance(value, np.integer):
        return int(value)
    return value


class OnlineSession:
    """Stateful streaming facade over an `ExpandedOnlineFilter` and a configuration.

    Args:
        fit: the fitted model (expanded engine).
        config: the project configuration (`data.source` column names, `data.covariates`, `data.clock`,
            `online`).
        initial: a prepared history for a warm start (the offline forward pass runs on it).
        lag: fixed-lag smoothing delay (default `online.lag`).
        nominal_dt: grid step in clock units for a cold start (`1` under a step clock; required otherwise
            unless `initial` is given).
        group: initial-distribution group of the stream (per-group fits).

    Raises:
        ConfigError: unsupported model or configuration (see the module docstring), a clock that needs a time
            column, or a missing `nominal_dt`.
    """

    def __init__(
        self,
        fit: FitResult,
        config: ProjectConfig,
        *,
        initial: PreparedDataset | None = None,
        lag: int | None = None,
        nominal_dt: float | None = None,
        group: str | None = None,
    ) -> None:
        """Validate the configuration against the model and start the filter."""
        from hsmmkit.model.build import required_own_lags

        data_cfg = config.data
        if data_cfg.preparation.observation_transform not in (None, "identity"):
            raise ConfigError(
                "online streaming supports only the identity observation transform",
                path="data.preparation.observation_transform",
            )
        if required_own_lags(config) > 0:
            raise ConfigError(
                "online streaming does not support own-lag (autoregressive) emissions", path="model"
            )
        self.fit = fit
        self.config = config
        self.obs_columns: list[str] = list(data_cfg.source.option("observation_columns", []) or [])
        if not self.obs_columns:
            raise ConfigError(
                "data.source.observation_columns is required", path="data.source.observation_columns"
            )
        self.time_column: str | None = data_cfg.source.option("time_column")
        unit = get_clock(data_cfg.clock).unit
        if self.time_column is None and unit != "step":
            raise ConfigError(
                "a clock in calendar units needs data.source.time_column", path="data.source.time_column"
            )
        if nominal_dt is None:
            nominal_dt = 1.0 if unit == "step" else None
        online_cfg = config.online
        alerts = [a.model_dump() for a in online_cfg.alerts] if online_cfg else []
        self.filter = ExpandedOnlineFilter(
            fit,
            group=group,
            nominal_dt=nominal_dt,
            lag=(online_cfg.lag if online_cfg else 0) if lag is None else lag,
            alerts=alerts,
        )
        self.state = self.filter.start(initial)
        self.events: list[AlertEvent] = []
        self.n_rows = int(self.state.t)
        self._cold = initial is None
        self._last_obs: np.ndarray | None = None
        self.cov_columns = [c.column or c.name for c in data_cfg.covariates]
        self._cov: OnlineCovariatePreparer | None = None
        if data_cfg.covariates:
            names = tuple(c.name for c in data_cfg.covariates)
            if tuple(fit.prep_stats.cov_names) != names:
                raise ConfigError(
                    f"the model was fitted with covariates {tuple(fit.prep_stats.cov_names)}, "
                    f"the configuration has {names}",
                    path="data.covariates",
                )
            entry: dict[str, list[int]] = {}
            for i, c in enumerate(data_cfg.covariates):
                for ep in c.use_in:
                    entry.setdefault(ep, []).append(i)
            self._cov = OnlineCovariatePreparer(
                fit.prep_stats,
                max_staleness=[c.max_staleness for c in data_cfg.covariates],
                entry_points=entry,
            )
            self._cov.push([np.nan] * len(names))  # the virtual start of the grid has no print
        self._obs_mean = fit.prep_stats.obs_mean
        self._obs_std = fit.prep_stats.obs_std

    # -- required columns ----------------------------------------------------------------------

    @property
    def required_columns(self) -> list[str]:
        """Every column a row must have: time (if configured), observations, covariates."""
        cols = [self.time_column] if self.time_column else []
        return [*cols, *self.obs_columns, *self.cov_columns]

    # -- pushing -------------------------------------------------------------------------------

    @staticmethod
    def _number(row: Mapping[str, Any], column: str) -> float:
        if column not in row:
            raise DataError(f"missing column {column!r}", path=column)
        value = row[column]
        if value is None or (isinstance(value, str) and not value.strip()):
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise DataError(f"column {column!r}: cannot read {value!r} as a number", path=column) from exc

    def push(self, row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        """Absorb one row.

        Returns:
            `{"t", "time", "filtered", "smoothed_lag", "events"}` (`filtered` is a list of `N` floats;
            `smoothed_lag` a list or `None` without a lag; `events` a list of `AlertEvent.to_dict()`); a
            warm-up row without a usable covariate gives `filtered: None` and no state change.

        Raises:
            DataError: a missing or unreadable column, a covariate without a print after the warm-up.
        """
        mapping: Mapping[str, Any] = (
            {str(k): v for k, v in row.items()} if isinstance(row, pd.Series) else row
        )
        y = np.array([self._number(mapping, c) for c in self.obs_columns])
        if self._obs_mean is not None and self._obs_std is not None:
            y = (y - self._obs_mean) / self._obs_std
        if self.time_column is not None:
            if self.time_column not in mapping:
                raise DataError(f"missing column {self.time_column!r}", path=self.time_column)
            time: Any = parse_times([mapping[self.time_column]])[0]
        else:
            time = self.n_rows
        z: dict[str, np.ndarray] | None = None
        warmup = False
        if self._cov is not None:
            prints = [self._number(mapping, c) for c in self.cov_columns]
            z = self._cov.z(self.n_rows)
            self._cov.push(prints)
            if any(np.isnan(v).any() for v in z.values()):
                if self._cold and int(self.state.t) == 0:
                    warmup = True
                else:
                    raise DataError("a covariate has no usable print for this row", path="data.covariates")
        self.n_rows += 1
        if warmup:
            return {
                "t": int(self.state.t),
                "time": _time_json(time),
                "filtered": None,
                "smoothed_lag": None,
                "events": [],
            }
        step = self.filter.update_from_time(self.state, time, y, z)
        self.state = step[0]
        filtered = np.asarray(step[1])
        smoothed = np.asarray(step[2]) if self.filter.lag > 0 else None
        events: list[AlertEvent] = step[-1] if self.filter.rules else []
        self.events.extend(events)
        self._last_obs = y
        return {
            "t": int(self.state.t),
            "time": _time_json(time),
            "filtered": filtered.tolist(),
            "smoothed_lag": None if smoothed is None else smoothed.tolist(),
            "events": [e.to_dict() for e in events],
        }

    def push_many(self, rows: pd.DataFrame) -> pd.DataFrame:
        """`push` every row of `rows`; one output row per processed step.

        Returns:
            A frame with `t`, `time`, `filtered_<i>` (and `smoothed_<i>` with a lag) and `n_events`.
        """
        out: list[dict[str, Any]] = []
        for _, row in rows.iterrows():
            result = self.push(row)
            if result["filtered"] is None:
                continue
            rec: dict[str, Any] = {"t": result["t"], "time": result["time"]}
            rec.update({f"filtered_{i}": v for i, v in enumerate(result["filtered"])})
            if result["smoothed_lag"] is not None:
                rec.update({f"smoothed_{i}": v for i, v in enumerate(result["smoothed_lag"])})
            rec["n_events"] = len(result["events"])
            out.append(rec)
        return pd.DataFrame(out)

    # -- persistence ---------------------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Write the filter state, the alert flags and the covariate history into the directory `path`."""
        out = save_state(self.state, path)
        blob = {
            "n_rows": self.n_rows,
            "cold": self._cold,
            "covariates": None if self._cov is None else self._cov.state_dict(),
        }
        (Path(out) / SESSION_FILE).write_text(json.dumps(blob), encoding="utf-8")
        return out

    def load(self, path: Path | str) -> None:
        """Restore a state written by `save` (resume); it must belong to the session's model."""
        src = Path(path)
        self.state = load_state(src, self.fit)
        try:
            blob = json.loads((src / SESSION_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(f"cannot read {src / SESSION_FILE}: {exc}", path=str(src)) from exc
        self.n_rows = int(blob["n_rows"])
        self._cold = bool(blob["cold"])
        if self._cov is not None and blob.get("covariates") is not None:
            self._cov.load_state_dict(blob["covariates"])

    # -- forecasting ---------------------------------------------------------------------------

    def forecast(
        self,
        horizon: int | None = None,
        *,
        n_sim: int | None = None,
        seed: int | None = None,
        history: PreparedDataset | None = None,
    ) -> ForecastResult:
        """Monte-Carlo forecast from the current state (defaults: `forecast.horizon`, `forecast.n_sim`).

        Without `history` a one-row history of the last processed row is used, which supports models without
        covariates; with covariates pass the prepared `history` ending at the current step.
        """
        cfg = self.config.forecast
        data = history if history is not None else self._one_row_history()
        return self.filter.forecast(
            self.state,
            cfg.horizon if horizon is None else horizon,
            cfg.n_sim if n_sim is None else n_sim,
            data,
            seed=self.config.project.seed if seed is None else seed,
            quantiles=cfg.quantiles,
        )

    def _one_row_history(self) -> PreparedDataset:
        from hsmmkit.data.contracts import PreparedDataset

        if self._cov is not None:
            raise ConfigError("forecasting a model with covariates needs history=", path="history")
        if self._last_obs is None or self.state.last_time is None:
            raise ConfigError("forecasting needs a processed row", path="state.t")
        import jax.numpy as jnp

        last = self.state.last_time
        times = (
            np.asarray([last]) if isinstance(last, np.datetime64) else np.asarray([int(last)], dtype=np.int64)
        )
        return PreparedDataset(
            times=times,
            observations=jnp.asarray(self._last_obs[None, :]),
            dt=jnp.asarray([self.filter.nominal_dt or 1.0]),
            mask=jnp.asarray(~np.isnan(self._last_obs[None, :])),
            cov_transition=None,
            cov_duration=None,
            cov_emission=None,
            cov_names={},
            clock=self.config.data.clock,
            nominal_dt=float(self.filter.nominal_dt or 1.0),
            prep_stats=self.fit.prep_stats,
            snapshot_ids=(),
        )
