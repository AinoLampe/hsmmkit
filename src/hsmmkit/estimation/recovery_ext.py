"""Extended parameter-recovery scenarios R7-R11 (E14-T06; Part II §16.4, Appendix G).

Pure functions (no pytest, no file I/O). Every scenario simulates with the deterministic generators of
`hsmmkit.presets.fixtures`, fits with the estimators named in the table and compares with the truth under the
tolerances of Appendix D / G (covariance entries 15 % relative, AR coefficients and regime means 2 standard
errors, mean sojourn 25 %, jump probabilities 0.15 absolute). A scenario passes when every *gating* category
holds in at least 4 of 5 seeds (`run_ext_suite`).

| id | data | model | estimators |
|---|---|---|---|
| R7 | `M = 3`, 50 x `T = 300` | full-covariance Gaussian, 3 regimes, NB `r = 2` | MAP, EM |
| R8 | counts, irregular calendar `dt` | Poisson, 2 regimes, gamma durations (explicit) | GEM |
| R9 | `M = 1`, `T = 5000` | MS-AR(2), mean-adjusted (Hamilton) | MAP |
| R10 | `M = 2`, 40 sessions, 30 % labels | left-to-right 4 regimes, per-state durations (explicit) | GEM, `init: labels` |
| R11 | `M = 1`, `T = 5000` | Gaussian diagonal, 4 configured / 3 true regimes | VB (exactly one pruned) |
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation import fit as run_fit
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.dists import unpack_lower
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.presets import fixtures as fx

if TYPE_CHECKING:
    from hsmmkit.config.schema import ProjectConfig
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.spec import ModelSpec

__all__ = ["EXT_SCENARIOS", "ExtCheck", "ExtReport", "ExtScenario", "ExtSuite", "run_ext", "run_ext_suite"]

SEEDS = (0, 1, 2, 3, 4)
COV_TOL = 0.15
SOJOURN_TOL = 0.25
JUMP_TOL = 0.15


@dataclass(frozen=True)
class ExtCheck:
    """One truth / estimate / tolerance comparison."""

    category: str
    name: str
    truth: float
    estimate: float
    rule: str
    passed: bool
    gating: bool = True

    def line(self) -> str:
        """`name: truth=.. estimate=.. (rule)`."""
        return f"{self.name}: truth={self.truth:.4g} estimate={self.estimate:.4g} ({self.rule})"


@dataclass(frozen=True)
class ExtReport:
    """One scenario, estimator and seed: the checks and the fit's health."""

    scenario: str
    estimator: str
    seed: int
    checks: tuple[ExtCheck, ...]
    finite: bool
    fit: FitResult | None = None

    @property
    def categories(self) -> dict[str, bool]:
        """`{category: all its checks pass}` over the gating checks."""
        out: dict[str, bool] = {}
        for c in self.checks:
            if c.gating:
                out[c.category] = out.get(c.category, True) and c.passed
        return out

    def failures(self) -> str:
        """The failing check lines."""
        lines = [c.line() for c in self.checks if c.gating and not c.passed]
        return "; ".join(lines) if lines else "no failing checks"


@dataclass(frozen=True)
class ExtScenario:
    """A scenario: its id, title, estimators and runner `(seed, estimator, quick) -> ExtReport`."""

    name: str
    title: str
    estimators: tuple[str, ...]
    run: Callable[[int, str, bool], ExtReport]


@dataclass(frozen=True)
class ExtSuite:
    """`run_ext_suite` output: the reports of one estimator and the per-category pass counts."""

    scenario: str
    estimator: str
    reports: tuple[ExtReport, ...]

    @property
    def category_counts(self) -> dict[str, tuple[int, int]]:
        """`{category: (seeds passing, seeds)}`."""
        cats = sorted({c for r in self.reports for c in r.categories})
        return {c: (sum(r.categories.get(c, False) for r in self.reports), len(self.reports)) for c in cats}

    @property
    def passed(self) -> bool:
        """Every gating category holds in at least 4 of 5 seeds (fewer seeds: in all but one)."""
        need = max(len(self.reports) - 1, 1)
        return all(k >= need for k, _ in self.category_counts.values())

    def summary(self) -> str:
        """One line per category plus the failures of the seeds that missed."""
        lines = [
            f"{self.scenario}/{self.estimator}: "
            + ", ".join(f"{c} {k}/{n}" for c, (k, n) in self.category_counts.items())
        ]
        lines += [f"  seed {r.seed}: {r.failures()}" for r in self.reports if not all(r.categories.values())]
        return "\n".join(lines)


# --- helpers ---------------------------------------------------------------------------------


def _raw(
    n: int, series: list[str], emission: dict[str, Any], duration: dict[str, Any], **model: Any
) -> dict[str, Any]:  # noqa: ANN401
    """A minimal configuration: `n` regimes, the emission and duration sections, extra `model` keys."""
    return {
        "schema_version": 1,
        "project": {"name": "recovery", "seed": 0, "output_dir": "runs"},
        "data": {
            "source": {"kind": "csv", "path": "unused.csv", "observation_columns": series},
            "clock": model.pop("clock", "steps"),
        },
        "model": {
            "n_states": n,
            "initial_state": "uniform",
            "emission": emission,
            "duration": {"covariates": [], **duration},
            "transition": {"mode": "jump", "order": 1, "covariates": []},
            **model,
        },
        "estimation": {"restarts": 1, "max_iter": 100, "init": ["rolling_kmeans"], "device": "cpu"},
        "forecast": {
            "horizon": 1,
            "n_sim": 50,
            "covariate_forecaster": {"type": "freeze"},
            "quantiles": [0.05, 0.5, 0.95],
        },
        "evaluation": {
            "metrics": ["log_score"],
            "backtest": {
                "scheme": "expanding",
                "initial_train_steps": 160,
                "refit_every_steps": 60,
                "horizons": [1],
                "embargo_steps": 0,
            },
        },
        "output": {"write": ["backtest"]},
    }


def _spec(
    raw: dict[str, Any], n_series: int, clock: str = "steps", nominal_dt: float = 1.0
) -> tuple[ProjectConfig, ModelSpec]:
    cfg = parse_project_config(raw)
    return cfg, build_model_spec_from(
        cfg, BuildInputs(n_series=n_series, cov_names={}, clock=clock, nominal_dt=nominal_dt)
    )


def _dataset(y: np.ndarray, dt: np.ndarray, clock: str = "steps", nominal_dt: float = 1.0) -> Any:  # noqa: ANN401
    from hsmmkit.estimation import recovery

    data = recovery._empty_prepared(y.shape[0], y, jnp.asarray(dt))
    return dataclasses.replace(data, clock=clock, nominal_dt=nominal_dt)


def _fit(spec: ModelSpec, data: Any, cfg: EstimationSection, seed: int, warm: Any = None) -> FitResult:  # noqa: ANN401
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return run_fit(spec, data, cfg, seed, warm)


def _finite(res: FitResult) -> bool:
    return bool(np.isfinite(res.loglik) and np.all(np.isfinite(np.asarray(res.uparams))))


def _rel_check(
    category: str, name: str, truth: float, est: float, tol: float, *, gating: bool = True
) -> ExtCheck:
    ok = bool(np.isfinite(est) and abs(est - truth) <= tol * abs(truth))
    return ExtCheck(category, name, float(truth), float(est), f"|rel err| <= {tol:.0%}", ok, gating)


def _abs_check(
    category: str, name: str, truth: float, est: float, tol: float, *, gating: bool = True
) -> ExtCheck:
    ok = bool(np.isfinite(est) and abs(est - truth) <= tol)
    return ExtCheck(category, name, float(truth), float(est), f"|err| <= {tol:g}", ok, gating)


def _jump_matrix(pi_rows: np.ndarray) -> np.ndarray:
    n = pi_rows.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        out[i, np.arange(n) != i] = pi_rows[i]
    return out


# --- R7: full-covariance Gaussian, batch, NB ---------------------------------------------------


def _r7(seed: int, estimator: str, quick: bool) -> ExtReport:
    tr = fx.R7_TRUTH
    n_seq, t_len = (10, 200) if quick else (50, 300)
    frame = fx.simulate_r7(seed, n_seq=n_seq, t_len=t_len)
    seqs = [
        _dataset(g[["y0", "y1", "y2"]].to_numpy(), np.ones(len(g)))
        for _, g in frame.groupby("sequence_id", sort=True)
    ]
    batch = SequenceBatch(tuple(seqs), (), tuple(str(i) for i in range(len(seqs))))
    emission = {
        "family": "gaussian",
        "covariance": "full",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {"mean": {"mode": "free", "init": [0.0, 0.0, 0.0]}},
    }
    duration = {
        "family": "negbin",
        "r": [2, 2, 2],
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [12.0, 12.0, 12.0]}},
    }
    _, spec = _spec(_raw(3, ["y0", "y1", "y2"], emission, duration), 3)
    est = EstimationSection.model_validate(
        {
            "method": estimator,
            "restarts": 3,
            "max_iter": 300,
            "init": ["rolling_kmeans", "random"],
            "device": "cpu",
        }
    )
    res = _fit(spec, batch, est, seed)
    th = {k: np.asarray(v) for k, v in res.params.items()}
    order = np.argsort(th["emission.mean"][:, 0])  # the truth is ordered by the first series' mean
    mean_hat = th["emission.mean"][order]
    chol_hat = np.asarray(unpack_lower(jnp.asarray(th["emission.chol"]), 3))[order]
    cov_hat = chol_hat @ np.swapaxes(chol_hat, -1, -2)
    cov_true = tr["chol"] @ np.swapaxes(tr["chol"], -1, -2)
    checks: list[ExtCheck] = []
    for i in range(3):
        for a in range(3):
            for b in range(a + 1):
                checks.append(
                    _rel_check(
                        "covariance", f"cov[{i}][{a},{b}]", cov_true[i, a, b], cov_hat[i, a, b], COV_TOL
                    )
                )
    for i in range(3):
        for a in range(3):
            checks.append(_abs_check("mean", f"mean[{i},{a}]", tr["mean"][i, a], mean_hat[i, a], 0.1))
    soj_hat = np.asarray(spec.duration.mean_sojourn(res.params, 1.0))[order]
    for i in range(3):
        checks.append(
            _rel_check("mean_sojourn", f"mean_sojourn[{i}]", tr["sojourn"][i], soj_hat[i], SOJOURN_TOL)
        )
    jump_hat = _jump_matrix(th["transition.pi_rows"])[np.ix_(order, order)]
    jump_true = _jump_matrix(tr["pi_rows"])
    for i in range(3):
        for j in range(3):
            if i != j:
                checks.append(_abs_check("jump", f"pi[{i},{j}]", jump_true[i, j], jump_hat[i, j], JUMP_TOL))
    return ExtReport("R7", estimator, seed, tuple(checks), _finite(res), res)


# --- R9: MS-AR(2), Hamilton mean-adjusted ------------------------------------------------------


def _r9(seed: int, estimator: str, quick: bool) -> ExtReport:
    from hsmmkit.data.contracts import ObservationSet
    from hsmmkit.data.prepare import Preparer
    from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
    from hsmmkit.estimation.objective import make_objective
    from hsmmkit.estimation.report import std_errors
    from hsmmkit.model.build import build_model_spec, required_own_lags
    from hsmmkit.statespace.builders import build_expanded

    tr = fx.R9_TRUTH
    frame = fx.simulate_r9(seed, 2000 if quick else 5000)
    emission = {
        "family": "msar",
        "ar": {"p": 2, "form": "mean_adjusted"},
        "params": {
            "mean": {"mode": "free", "init": [-0.3, 0.8], "constraint": "increasing"},
            "scale": {"mode": "free", "init": [1.0, 1.0]},
        },
    }
    duration = {
        "family": "geometric",
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [12.0, 15.0]}},
    }
    cfg = parse_project_config(_raw(2, ["x"], emission, duration))
    times = pd.bdate_range("2000-01-03", periods=len(frame)).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, frame[["y"]].to_numpy(), "memory", "synthetic")
    data = (
        Preparer(cfg.data, own_lags=required_own_lags(cfg))
        .fit(obs, [], train_end=times[-1])
        .transform(obs, [])
    )
    spec = build_model_spec(cfg, data)
    est = EstimationSection.model_validate(
        {"method": estimator, "restarts": 3, "max_iter": 300, "init": ["rolling_kmeans"], "device": "cpu"}
    )
    res = _fit(spec, data, est, seed)
    manager = ParamManager(spec.param_specs)
    objective = make_objective(
        spec, build_expanded(spec), ExpandedHMMEngine(), manager, data, use_priors=False
    )
    se = std_errors(objective, res.uparams, manager)
    th = {k: np.asarray(v) for k, v in res.params.items()}
    checks: list[ExtCheck] = []
    for key, truth, cat in (("emission.ar", tr["phi"], "ar"), ("emission.mean", tr["mean"], "mean")):
        est_v = th[key].ravel()
        sd = np.asarray(se[key]).ravel() if se is not None else np.full(truth.shape, np.nan)
        for i, (t, e, s_) in enumerate(zip(truth, est_v, sd, strict=True)):
            ok = bool(np.isfinite(s_) and abs(e - t) <= 2.0 * s_)
            checks.append(ExtCheck(cat, f"{key}[{i}]", float(t), float(e), f"|err| <= 2 SE ({s_:.3g})", ok))
    soj = 1.0 / np.asarray(spec.duration.advance_prob_t(res.params, jnp.asarray(1.0), None))
    for i in range(2):
        checks.append(
            _rel_check("mean_sojourn", f"mean_sojourn[{i}]", tr["sojourn"][i], float(soj[i]), SOJOURN_TOL)
        )
    return ExtReport("R9", estimator, seed, tuple(checks), _finite(res), res)


# --- R11: VB with one superfluous regime ------------------------------------------------------


def _r11(seed: int, estimator: str, quick: bool) -> ExtReport:
    tr = fx.R11_TRUTH
    frame = fx.simulate_r11(seed, 2500 if quick else 5000)
    data = _dataset(frame[["y"]].to_numpy(), np.ones(len(frame)))
    emission = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {
            "mean": {"mode": "free", "init": [-2.5, -0.5, 0.5, 2.5]},
            "scale": {"mode": "free", "init": [0.8] * 4},
        },
    }
    duration = {
        "family": "geometric",
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [20.0] * 4}},
    }
    _, spec = _spec(_raw(4, ["y"], emission, duration), 1)
    est = EstimationSection.model_validate(
        {"method": "vb", "restarts": 0, "max_iter": 300, "tol_f": 1e-10, "device": "cpu"}
    )
    res = _fit(spec, data, est, 0, ParamManager(spec.param_specs).init_u())
    keep = [i for i in range(4) if i not in res.pruned_states]
    mean_hat = np.sort(np.asarray(res.params["emission.mean"]).ravel()[keep])
    checks = [
        ExtCheck(
            "pruned",
            "n_pruned",
            1.0,
            float(len(res.pruned_states)),
            "exactly one regime pruned",
            len(res.pruned_states) == 1,
        )
    ]
    if len(keep) == 3:
        for i in range(3):
            checks.append(_abs_check("mean", f"mean[{i}]", tr["mean"][i], float(mean_hat[i]), 0.15))
    return ExtReport("R11", "vb", seed, tuple(checks), _finite(res), res)


# --- R8: Poisson counts, gamma durations, irregular calendar grid -----------------------------


def _r8(seed: int, estimator: str, quick: bool) -> ExtReport:
    tr = fx.R8_TRUTH
    frame = fx.simulate_r8(seed, 3000 if quick else 8000)
    dt = frame["dt"].to_numpy()
    nominal = float(np.median(dt))
    data = _dataset(frame[["y"]].to_numpy(), dt, clock="calendar", nominal_dt=nominal)
    emission = {
        "family": "poisson",
        "scale_by_dt": True,
        "params": {"rate": {"mode": "free", "init": [1500.0, 3500.0]}},  # per year
    }
    duration = {
        "family": "gamma",
        "params": {
            "shape": {"mode": "free", "init": [2.0, 2.0]},
            "scale": {"mode": "free", "init": [5.0 * nominal, 5.0 * nominal]},
        },
    }
    _, spec = _spec(
        _raw(2, ["y"], emission, duration, engine="explicit", d_max=120, clock="calendar"),
        1,
        "calendar",
        nominal,
    )
    est = EstimationSection.model_validate(
        {
            "method": estimator,
            "restarts": 0,
            "max_iter": 300,
            "tol_f": 1e-9,
            "use_priors": False,
            "device": "cpu",
        }
    )
    res = _fit(spec, data, est, seed, ParamManager(spec.param_specs).init_u())
    rate_hat = np.asarray(res.params["emission.rate"]).ravel()
    order = np.argsort(rate_hat)
    per_day = rate_hat[order] * nominal
    soj = np.asarray(spec.duration.mean_sojourn(res.params, nominal))[order] / nominal
    soj_true = tr["shape"] * tr["scale_steps"]
    checks: list[ExtCheck] = []
    for i in range(2):
        checks.append(_rel_check("rate", f"rate[{i}]", tr["rate_per_step"][i], float(per_day[i]), 0.1))
        checks.append(
            _rel_check("mean_sojourn", f"mean_sojourn[{i}]", soj_true[i], float(soj[i]), SOJOURN_TOL)
        )
    return ExtReport("R8", estimator, seed, tuple(checks), _finite(res), res)


# --- R10: left-to-right topology, per-state durations, labels -------------------------------


def _r10(seed: int, estimator: str, quick: bool) -> ExtReport:
    tr = fx.R10_TRUTH
    frame = fx.simulate_r10(seed, 30 if quick else 60)
    seqs = []
    for _, g in frame.groupby("sequence_id", sort=True):
        data = _dataset(g[["y0", "y1"]].to_numpy(), np.ones(len(g)))
        lab = g["label"].to_numpy()
        allowed = np.ones((len(g), 4), dtype=bool)
        known = np.isfinite(lab)
        allowed[known] = False
        allowed[np.flatnonzero(known), lab[known].astype(int)] = True
        seqs.append(dataclasses.replace(data, labels=jnp.asarray(allowed)))
    batch = SequenceBatch(tuple(seqs), (), tuple(str(i) for i in range(len(seqs))))
    emission = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {
            "mean": {"mode": "free", "init": [0.0, 0.0]},
            "scale": {"mode": "free", "init": [1.0, 1.0]},
        },
    }
    duration = {
        "per_state": [
            {
                "family": "gamma",
                "params": {"shape": {"mode": "free", "init": 2.0}, "scale": {"mode": "free", "init": 4.0}},
            },
            {
                "family": "gamma",
                "params": {"shape": {"mode": "free", "init": 2.0}, "scale": {"mode": "free", "init": 4.0}},
            },
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 10.0}}},
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 10.0}}},
        ]
    }
    topology = {"matrix": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]}
    raw = _raw(4, ["y0", "y1"], emission, duration, engine="explicit", d_max=60, topology=topology)
    raw["model"]["initial_state"] = "free"
    _, spec = _spec(raw, 2)
    est = EstimationSection.model_validate(
        {
            "method": estimator,
            "restarts": 1,
            "max_iter": 300,
            "tol_f": 1e-9,
            "init": ["labels"],
            "use_priors": False,
            "device": "cpu",
        }
    )
    res = _fit(spec, batch, est, seed)
    mean_hat = np.asarray(res.params["emission.mean"]).reshape(4, 2)
    sd_hat = np.asarray(res.params["emission.scale"]).reshape(4, -1)[:, 0]
    soj = np.asarray(spec.duration.mean_sojourn(res.params, 1.0))
    soj_true = [tr["shape"][0] * tr["scale"][0], tr["shape"][1] * tr["scale"][1], tr["geometric_mean"]]
    checks: list[ExtCheck] = []
    for k in range(4):
        for a in range(2):
            checks.append(_abs_check("mean", f"mean[{k},{a}]", tr["mean"][k, a], float(mean_hat[k, a]), 0.15))
        checks.append(_rel_check("sigma", f"sigma[{k}]", tr["sd"], float(sd_hat[k]), 0.1))
    for k in range(3):
        checks.append(
            _rel_check("mean_sojourn", f"mean_sojourn[{k}]", soj_true[k], float(soj[k]), SOJOURN_TOL)
        )
    return ExtReport("R10", estimator, seed, tuple(checks), _finite(res), res)


EXT_SCENARIOS: dict[str, ExtScenario] = {
    "R7": ExtScenario("R7", "full-covariance Gaussian, batch 50 x 300, NB r = 2", ("map", "em"), _r7),
    "R8": ExtScenario("R8", "Poisson, gamma durations, irregular calendar dt", ("gem",), _r8),
    "R9": ExtScenario("R9", "MS-AR(2), Hamilton mean-adjusted", ("map",), _r9),
    "R10": ExtScenario("R10", "left-to-right 4 regimes, per-state durations, 30 % labels", ("gem",), _r10),
    "R11": ExtScenario("R11", "VB, 4 configured / 3 true regimes, one pruned", ("vb",), _r11),
}


def run_ext(scenario: str, seed: int, estimator: str | None = None, *, quick: bool = False) -> ExtReport:
    """One report of `scenario` (R7-R11) at `seed` with `estimator` (default: the scenario's first)."""
    from hsmmkit.utils.errors import ConfigError

    if scenario not in EXT_SCENARIOS:
        raise ConfigError(
            f"unknown scenario {scenario!r}; available: {sorted(EXT_SCENARIOS)}", path="scenario"
        )
    sc = EXT_SCENARIOS[scenario]
    est = estimator or sc.estimators[0]
    if est not in sc.estimators:
        raise ConfigError(f"{scenario} runs with {list(sc.estimators)}, not {est!r}", path="estimator")
    return sc.run(seed, est, quick)


def run_ext_suite(
    scenario: str, estimator: str | None = None, seeds: Sequence[int] = SEEDS, *, quick: bool = False
) -> ExtSuite:
    """`run_ext` over `seeds`; the suite passes when every gating category holds in >= 4 of 5 seeds."""
    est = estimator or EXT_SCENARIOS[scenario].estimators[0]
    return ExtSuite(scenario, est, tuple(run_ext(scenario, s, est, quick=quick) for s in seeds))
