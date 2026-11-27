"""Grid model selection over `N`, `r`, emission / duration families and estimators (E14-T02; Part II §14.12).

`evaluation.grid` lists the axes; the cells are their Cartesian product (an absent axis keeps the base value).
Every cell is the base configuration with overrides, validated on its own: a cell that does not validate or
cannot be built (for example `gamma` durations on the expanded engine) is **skipped** and keeps its reason in
`skipped_reason`. Per cell the model is fitted with the cell's estimator (restarts from the configuration) and
scored by the information criteria (`loglik`, `n_free`, `aic`, `bic`, `icl`), the cross-validated log score
and CRPS when `evaluation.cv` is set, the smallest regime occupancy and the spread of the restart objectives.

* Changing `n_states` resizes the per-regime lists of the base configuration (`init`, `values`,
  `init_mean_sojourn_steps`, `r`) by linear interpolation between their ends, so a base with two initial means
  yields a grid cell with three or four spread initial means.
* A new emission / duration family starts from the default parameters of `FAMILY_PARAMS` (the primary initial
  sojourn of the base duration is carried over); families without an entry are skipped.
* Seeds: cell `i` fits with `derive_seed(seed, PURPOSE_GRID, i)`, so the table is independent of the execution
  order. `evaluation.grid.parallel` runs the cells in a process pool (`spawn`); the table is identical.
* Recommendation: the lowest BIC; ties (within 1e-9) go to the highest cross-validated log score.
"""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hsmmkit.config.schema import ProjectConfig, parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation import fit as fit_model
from hsmmkit.inference.decode import decode
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.selection.cv import cross_validate, make_splitter
from hsmmkit.utils.errors import ConfigError, HsmmkitError
from hsmmkit.utils.rng import PURPOSE_GRID, derive_seed

if TYPE_CHECKING:
    from hsmmkit.data.contracts import CovariateSeries, ObservationSet

__all__ = ["AXES", "COLUMNS", "FAMILY_PARAMS", "GridCell", "make_cells", "recommend", "run_grid"]

AXES = ("n_states", "r", "emission", "duration", "estimator")
COLUMNS = (
    "cell",
    *AXES,
    "loglik",
    "n_free",
    "aic",
    "bic",
    "icl",
    "cv_log_score",
    "cv_crps",
    "min_occupancy",
    "objective_spread",
    "wall_s",
    "converged",
    "n_iter",
    "skipped_reason",
    "recommended",
)

FAMILY_PARAMS: dict[str, dict[str, Any]] = {
    "gaussian": {"mean": {"mode": "free"}, "scale": {"mode": "free"}},
    "student_t": {"m": {"mode": "free"}, "s": {"mode": "free"}, "nu": {"mode": "free"}},
    "poisson": {"rate": {"mode": "free"}},
    "geometric": {"rate": {"mode": "free"}},
    "negbin": {"rate": {"mode": "free"}},
    "gamma": {"shape": {"mode": "free"}, "scale": {"mode": "free"}},
    "weibull": {"shape": {"mode": "free"}, "scale": {"mode": "free"}},
    "lognormal": {"mu_log": {"mode": "free"}, "sd_log": {"mode": "free"}},
}
_PRIMARY = {"gaussian": "mean", "student_t": "m", "poisson": "rate"}
_LISTS = ("init", "values", "init_mean_sojourn_steps")


@dataclass(frozen=True)
class GridCell:
    """One cell: its index, the axis values and the raw configuration (or the reason it is invalid)."""

    index: int
    axes: dict[str, Any]
    raw: dict[str, Any] | None
    skipped_reason: str = ""


def _resize(values: list[float], n: int) -> list[float]:
    """`values` interpolated linearly between its ends to length `n`."""
    if len(values) == n or len(values) < 2:
        return list(values) if len(values) == n else [float(values[0])] * n
    return [float(v) for v in np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(values)), values)]


def _resize_lists(node: Any, old_n: int, new_n: int) -> Any:  # noqa: ANN401
    """Resize every per-regime list of `node` (a nested config dict) from `old_n` to `new_n` entries."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if (
                key in _LISTS
                and isinstance(value, list)
                and len(value) == old_n
                and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
            ):
                out[key] = _resize(value, new_n)
            else:
                out[key] = _resize_lists(value, old_n, new_n)
        return out
    if isinstance(node, list):
        return [_resize_lists(v, old_n, new_n) for v in node]
    return node


def _apply(raw: dict[str, Any], axes: dict[str, Any]) -> dict[str, Any]:
    """The base configuration with the axis values of one cell applied (not yet validated)."""
    out = copy.deepcopy(raw)
    model = out["model"]
    old_n = int(model["n_states"])
    n = int(axes.get("n_states", old_n))
    if n != old_n:
        out["model"] = model = _resize_lists(model, old_n, n)
        model["n_states"] = n
        base_r = model["duration"].get("r")
        if isinstance(base_r, list) and len(base_r) == old_n:
            model["duration"]["r"] = [round(v) for v in _resize([float(v) for v in base_r], n)]
        if isinstance(model.get("topology"), dict):  # a structural topology does not survive a change of N
            raise ConfigError(
                "n_states cannot change under an explicit topology matrix", path="model.topology"
            )
    if "emission" in axes and axes["emission"] != model["emission"].get("family"):
        family = axes["emission"]
        model["emission"] = {"family": family, "params": copy.deepcopy(FAMILY_PARAMS.get(family, {}))}
        if family not in FAMILY_PARAMS:
            raise ConfigError(
                f"no default parameters for emission family {family!r}", path="evaluation.grid.emission"
            )
    if "duration" in axes and axes["duration"] != model["duration"].get("family"):
        family = axes["duration"]
        if family not in FAMILY_PARAMS:
            raise ConfigError(
                f"no default parameters for duration family {family!r}", path="evaluation.grid.duration"
            )
        base = model["duration"]
        sojourn = None
        for p in base.get("params", {}).values():
            if isinstance(p, dict) and "init_mean_sojourn_steps" in p:
                sojourn = p["init_mean_sojourn_steps"]
        params = copy.deepcopy(FAMILY_PARAMS[family])
        primary = {
            "negbin": "rate",
            "geometric": "rate",
            "gamma": "scale",
            "weibull": "scale",
            "lognormal": "mu_log",
        }
        if sojourn is not None:
            params[primary.get(family, "rate")]["init_mean_sojourn_steps"] = sojourn
        model["duration"] = {"family": family, "params": params}
        if family == "negbin":
            model["duration"]["r"] = [2] * n
    if "r" in axes:
        if model["duration"].get("family") != "negbin":
            raise ConfigError("the r axis needs negbin durations", path="evaluation.grid.r")
        r = axes["r"]
        r_list = list(r) if isinstance(r, list | tuple) else [int(r)] * n
        if len(r_list) != n:
            raise ConfigError(f"r has {len(r_list)} entries for {n} regimes", path="evaluation.grid.r")
        model["duration"]["r"] = r_list
    if "estimator" in axes:
        out["estimation"]["method"] = axes["estimator"]
    return out


def make_cells(cfg: ProjectConfig) -> list[GridCell]:
    """The Cartesian product of `evaluation.grid` as validated cells (invalid ones carry a reason).

    Raises:
        ConfigError: `evaluation.grid` is absent or empty.
    """
    grid = cfg.evaluation.grid
    if grid is None:
        raise ConfigError("evaluation.grid is required for a grid search", path="evaluation.grid")
    values: dict[str, list[Any]] = {}
    if grid.n_states:
        values["n_states"] = list(grid.n_states)
    if grid.r or grid.r_same_for_all:
        values["r"] = [list(r) for r in grid.r] + [int(v) for v in grid.r_same_for_all]
    if grid.emission:
        values["emission"] = list(grid.emission)
    if grid.duration:
        values["duration"] = list(grid.duration)
    if grid.estimator:
        values["estimator"] = list(grid.estimator)
    if not values:
        raise ConfigError("evaluation.grid has no axis", path="evaluation.grid")
    base = cfg.model_dump(mode="json", exclude_none=False)
    base["evaluation"]["grid"] = None
    base["evaluation"]["cv"] = base["evaluation"].get("cv")
    cells: list[GridCell] = []
    names = list(values)
    for index, combo in enumerate(itertools.product(*values.values())):
        axes = dict(zip(names, combo, strict=True))
        try:
            raw = _apply(base, axes)
            parse_project_config(raw, type(cfg))
        except (ConfigError, KeyError, TypeError) as exc:
            cells.append(GridCell(index, axes, None, f"{type(exc).__name__}: {exc}"))
            continue
        cells.append(GridCell(index, axes, raw))
    return cells


def _prepare(
    cfg: ProjectConfig,
    obs: ObservationSet,
    covs: Sequence[CovariateSeries],
    preparer_cls: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """The prepared dataset (or batch) of `cfg`: statistics from the whole sample (in-sample criteria)."""
    from hsmmkit.model.build import required_own_lags, required_series_support

    prep = preparer_cls(
        cfg.data,
        own_lags=required_own_lags(cfg),
        series_support=required_series_support(cfg),
        n_regimes=cfg.model.n_states,
    )
    end = obs.values.shape[0] - 1 if obs.times is None else np.max(obs.require_times())
    fitted = prep.fit(obs, covs, end)
    multi = obs.sequence_ids is not None and np.unique(np.asarray(obs.sequence_ids)).shape[0] > 1
    return fitted.transform_batch(obs, covs) if multi else fitted.transform(obs, covs)


def _run_cell(
    cell_raw: dict[str, Any],
    cfg_cls: type[ProjectConfig],
    obs: ObservationSet,
    covs: Sequence[CovariateSeries],
    seed: int,
    index: int,
    preparer_cls: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Fit and score one valid cell; a failing fit is reported as a skipped cell."""
    cfg = parse_project_config(cell_raw, cfg_cls)
    started = time.perf_counter()
    cell_seed = derive_seed(seed, PURPOSE_GRID, index)
    try:
        data = _prepare(cfg, obs, covs, preparer_cls)
        spec = build_model_spec(cfg, data)
        fit = fit_model(spec, data, cfg.estimation, cell_seed)
    except (HsmmkitError, NotImplementedError) as exc:  # a combination the engines cannot build
        return {"skipped_reason": f"{type(exc).__name__}: {exc}"}
    n_free = ParamManager(spec.param_specs).n_free
    row: dict[str, Any] = {
        "loglik": fit.loglik,
        "n_free": n_free,
        "converged": fit.converged,
        "n_iter": fit.n_iter,
    }
    objectives = np.sort(fit.restarts["objective"].to_numpy(dtype=np.float64))
    objectives = objectives[np.isfinite(objectives)]
    row["objective_spread"] = float(objectives[-1] - objectives[0]) if objectives.size > 1 else 0.0
    if isinstance(data, SequenceBatch):
        n_obs = int(sum(int(np.asarray(s.mask).all(axis=-1).sum()) for s in data.sequences))
        row["aic"] = 2.0 * n_free - 2.0 * fit.loglik
        row["bic"] = n_free * float(np.log(max(n_obs, 1))) - 2.0 * fit.loglik
        row["icl"] = float("nan")
        posts = decode(fit, data)
        occupancy = np.mean(np.concatenate([np.asarray(p.smoothed) for p in posts]), axis=0)
    else:
        post = decode(fit, data)
        row["aic"], row["bic"], row["icl"] = post.aic, post.bic, post.icl
        occupancy = np.mean(np.asarray(post.smoothed), axis=0)
    row["min_occupancy"] = float(np.min(occupancy))
    row["cv_log_score"] = row["cv_crps"] = float("nan")
    if cfg.evaluation.cv is not None:
        from hsmmkit.model.build import required_own_lags, required_series_support

        def factory() -> Any:  # noqa: ANN401
            return preparer_cls(
                cfg.data,
                own_lags=required_own_lags(cfg),
                series_support=required_series_support(cfg),
                n_regimes=cfg.model.n_states,
            )

        result = cross_validate(cfg, (obs, covs), make_splitter(cfg), cell_seed, factory)
        by_metric = result.summary.groupby("metric")["mean"].first()
        row["cv_log_score"] = float(by_metric.get("log_score", float("nan")))
        row["cv_crps"] = float(by_metric.get("crps", float("nan")))
    row["wall_s"] = time.perf_counter() - started
    row["skipped_reason"] = ""
    return row


def recommend(table: pd.DataFrame) -> int | None:
    """Index (`cell`) of the recommended cell: the lowest BIC, ties to the highest CV log score."""
    ok = table[(table["skipped_reason"] == "") & np.isfinite(table["bic"])]
    if ok.empty:
        return None
    best = ok["bic"].min()
    tied = ok[ok["bic"] <= best + 1e-9]
    if len(tied) > 1 and tied["cv_log_score"].notna().any():
        tied = tied.sort_values("cv_log_score", ascending=False, kind="stable")
    return int(tied.iloc[0]["cell"])


def run_grid(
    cfg: ProjectConfig,
    obs: ObservationSet,
    covs: Sequence[CovariateSeries],
    seed: int,
    preparer_cls: Any,  # noqa: ANN401
    *,
    on_cell: Callable[[int, dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every cell of `evaluation.grid`; returns the table (`COLUMNS`).

    Raises:
        ConfigError: no grid, or an empty one.
    """
    cells = make_cells(cfg)
    grid = cfg.evaluation.grid
    assert grid is not None
    results: dict[int, dict[str, Any]] = {}
    valid = [c for c in cells if c.raw is not None]
    if grid.parallel is not None and len(valid) > 1:
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=grid.parallel.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                c.index: pool.submit(_run_cell, c.raw, type(cfg), obs, covs, seed, c.index, preparer_cls)
                for c in valid
                if c.raw is not None
            }
            for index, future in futures.items():
                results[index] = future.result()
    else:
        for c in valid:
            assert c.raw is not None
            results[c.index] = _run_cell(c.raw, type(cfg), obs, covs, seed, c.index, preparer_cls)
            if on_cell is not None:
                on_cell(c.index, results[c.index])
    rows: list[dict[str, Any]] = []
    for c in cells:
        row: dict[str, Any] = {"cell": c.index, **{a: c.axes.get(a) for a in AXES}}
        if c.raw is None:
            row["skipped_reason"] = c.skipped_reason
        else:
            row.update(results[c.index])
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in table.columns:
            table[col] = (
                np.nan
                if col not in ("skipped_reason", "recommended")
                else ("" if col == "skipped_reason" else False)
            )
    table["r"] = table["r"].map(
        lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    )
    table["skipped_reason"] = table["skipped_reason"].fillna("").astype(str)
    best = recommend(table)
    table["recommended"] = table["cell"] == best if best is not None else False
    return table[list(COLUMNS)].reset_index(drop=True)


def render_grid_markdown(table: pd.DataFrame) -> str:
    """`grid.md`: the table with the best value of each criterion in bold and the recommendation rule."""
    ok = table[table["skipped_reason"] == ""]
    best = {
        "aic": ok["aic"].idxmin() if not ok.empty else None,
        "bic": ok["bic"].idxmin() if not ok.empty else None,
        "icl": ok["icl"].idxmin() if ok["icl"].notna().any() else None,
        "cv_log_score": ok["cv_log_score"].idxmax() if ok["cv_log_score"].notna().any() else None,
        "cv_crps": ok["cv_crps"].idxmin() if ok["cv_crps"].notna().any() else None,
    }
    cols = [c for c in COLUMNS if c not in ("skipped_reason", "recommended")]
    lines = [
        "# Grid model selection",
        "",
        "Recommendation rule: the lowest BIC; ties (within 1e-9) go to the highest cross-validated log "
        "score. "
        "The best value of each criterion is **bold**; skipped cells carry their reason below the table.",
        "",
        "| " + " | ".join([*cols, "recommended"]) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for idx, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            text = (
                "-"
                if v is None or (isinstance(v, float) and not np.isfinite(v))
                else (f"{v:.6g}" if isinstance(v, float | np.floating) else str(v))
            )
            cells.append(f"**{text}**" if best.get(c) == idx and row["skipped_reason"] == "" else text)
        lines.append("| " + " | ".join([*cells, "yes" if row["recommended"] else ""]) + " |")
    skipped = table[table["skipped_reason"] != ""]
    if not skipped.empty:
        lines += ["", "## Skipped cells", ""]
        lines += [f"- cell {int(r['cell'])}: {r['skipped_reason']}" for _, r in skipped.iterrows()]
    return "\n".join(lines) + "\n"
