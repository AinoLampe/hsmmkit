"""Parameter-recovery suite (E5-T05a): simulate from a known `theta*`, fit, check recovery tolerances.

CLAUDE.md §3.2 places the harness here (not `tests/recovery/harness.py`): it is pure (no pytest, no
file I/O), so `tests/recovery/test_e5_t05a_recovery.py` (`-m slow`) and E5-T05b's `selftest` CLI both
import it directly. Six scenarios (concept Appendix D / `E5-T05a` ticket §Context), `N=3`, daily
(`clock="trading"`) grid, `constraint="increasing"` on `emission.scale` in every scenario (an
identifiability guard, so the MAP fit never needs post-hoc relabelling). Models are assembled by hand
(`tests/helpers/objective_fixtures.py`'s pattern) since `model/build.py` (CLAUDE.md §3.3) builds from a
project *config*, not from arbitrary truth parameters.

Tolerances (concept Appendix D, quoted verbatim, never loosened): sigma <= 10% rel; mu <= 2 SE (when
free); mean sojourn <= 25% rel; jump probabilities <= 0.15 abs; coefficients sign-correct and <= 2 SE.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.config.schema import EstimationSection
from regimekit.data.contracts import PreparedDataset, PrepStats
from regimekit.engine.base import InferenceEngine
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.estimation.map import fit
from regimekit.estimation.objective import make_objective
from regimekit.estimation.report import std_errors
from regimekit.estimation.results import FitResult
from regimekit.inference.simulate import simulate_dataset
from regimekit.model.duration.gamma import GammaDuration
from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.model.params.manager import ParamManager, Params
from regimekit.model.params.priors import DirichletPrior, NormalPrior
from regimekit.model.params.spec import Mode, ParamSpec
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition, build_memory_layout
from regimekit.statespace.builders import build_expanded, build_explicit
from regimekit.statespace.index import StateSpace
from regimekit.utils.clock import get_clock, parse_clock_options
from regimekit.utils.rng import key_for

if TYPE_CHECKING:
    from regimekit.model.duration.base import DurationFamily
    from regimekit.model.emission.base import EmissionFamily
    from regimekit.model.transition import TransitionModel

# --- truth values shared by every scenario (concept Appendix D worked example) -----------------

RECOVERY_CLOCK = "trading"  # daily grid, parameters in years (clock registry)


@functools.cache
def nominal_dt() -> float:
    """The daily step of the recovery scenarios, taken from the `RECOVERY_CLOCK` clock (E10-T03)."""
    clock = get_clock(RECOVERY_CLOCK)
    return float(clock.dt(None, 1, parse_clock_options(RECOVERY_CLOCK, None))[0])


N_STATES = 3
SIGMA_STAR = np.array([0.10, 0.18, 0.35])
MU_STAR = np.array([0.10, 0.00, -0.20])
MEAN_SOJOURN_DAYS = np.array([60.0, 120.0, 30.0])  # steps, one bar/day
R_PHASES = (3, 3, 2)
# Row r = current regime r's jump distribution over targets, in ascending-target-index order
# (`regimekit.model.transition._target_order`'s convention: first entry is the reference target).
PI_ROWS_TARGETS = np.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
TRANS_COEF_STAR = np.array([0.5, -0.5])  # two transition covariates
# One duration covariate, per state. Owner decision (E5-T05a): the ticket's (0.3, -0.3, 0.2) has the same
# size as the standard error at T=20000 (~0.18), so 'sign-correct' fails in ~half the 5-seed runs even for a
# perfect estimator; doubling the effect makes the check attainable and exercises the same code path.
DURATION_COEF_STAR = np.array([0.6, -0.6, 0.4])

SIGMA_REL_TOL = 0.10
SE_MULT = 2.0
MEAN_SOJOURN_REL_TOL = 0.25
JUMP_ABS_TOL = 0.15

RESTARTS = 4
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
_MIN_PASS = 4
_SIM_PURPOSE = 1
_COV_PURPOSE = 2


DEFAULT_RECOVERY_T = 20000


def _default_t_len() -> int:
    """`REGIMEKIT_RECOVERY_T` env var, default `DEFAULT_RECOVERY_T` = 20000 (the sizing knob of AC3).

    At `T=5000` the truth gives only ~64 regime visits per path, so even an oracle that reads the true
    regime path meets the sojourn and jump tolerances together in only half of the seeds; at `T=20000`
    (~256 visits) it does in ~99 % (docs/guide/validation.md).
    """
    return int(os.environ.get("REGIMEKIT_RECOVERY_T", str(DEFAULT_RECOVERY_T)))


# --- covariate generation (synthetic exogenous AR(1) series, standardized) ----------------------


def _ar1_covariates(t_len: int, k: int, seed: int, *, phi: float = 0.6) -> np.ndarray:
    """`(t_len, k)` standardized AR(1) covariates; NumPy RNG (recovery harness, not a core jit path)."""
    rng = np.random.default_rng(seed)
    out = np.empty((t_len, k))
    for j in range(k):
        eps = rng.normal(size=t_len)
        x = np.empty(t_len)
        x[0] = eps[0]
        for t in range(1, t_len):
            x[t] = phi * x[t - 1] + eps[t]
        out[:, j] = (x - x.mean()) / x.std()
    return out


# --- transition-row helpers -----------------------------------------------------------------


def _a_from_pi_row(row: np.ndarray) -> np.ndarray:
    """Target-order probabilities `[ref, nonref...]` -> logits `log(nonref / ref)` (z=0 baseline)."""
    ref, *nonref = row
    return np.asarray(np.log(np.asarray(nonref, dtype=np.float64) / ref), dtype=np.float64)


# --- scenario definition ---------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A recovery scenario: how to build `(spec, theta_star)` and which covariates to simulate."""

    name: str
    order: int = 1
    duration: Literal["negbin", "gamma"] = "negbin"
    t_scale: float = 1.0
    """Multiplier on the default `T` (`order2`: 2.5, its six memory rows see ~1/6 of the transitions each)."""
    cov_transition: bool = False
    cov_duration: bool = False
    mu_fixed: bool = False
    d_max: int = 20
    report_only: tuple[str, ...] = ()
    """Tolerance categories that are reported but not part of the pass criterion (owner decision)."""


def _build_spec_and_truth(scenario: Scenario) -> tuple[ModelSpec, Params]:
    n = N_STATES
    specs: list[ParamSpec] = []
    theta: Params = {}

    # emission: GBM, sigma always free + increasing (identifiability guard); mu free unless scenario 6.
    mu_mode: Mode = "fixed" if scenario.mu_fixed else "free"
    specs.append(
        ParamSpec(
            name="emission.mean",
            shape=(n, 1),
            domain="real",
            mode=mu_mode,
            value=MU_STAR[:, None] if scenario.mu_fixed else None,
            init=None if scenario.mu_fixed else np.zeros((n, 1)),
        )
    )
    specs.append(
        ParamSpec(
            name="emission.scale",
            shape=(n,),
            domain="positive",
            mode="free",
            init=np.linspace(0.05, 0.5, n),
            constraint="increasing",
        )
    )
    theta["emission.mean"] = jnp.asarray(MU_STAR[:, None])
    theta["emission.scale"] = jnp.asarray(SIGMA_STAR)
    emis = GaussianFamily(scale_by_dt=True, ito_correction=True)

    # duration: phase negbin (expanded engine) or explicit gamma (explicit engine).
    dur: NegBinDuration | GammaDuration
    if scenario.duration == "negbin":
        dur_cov_idx = (0,) if scenario.cov_duration else ()
        dur = NegBinDuration(r=R_PHASES, cov_idx=dur_cov_idx)
        specs.append(
            ParamSpec(name="duration.rate", shape=(n,), domain="positive", mode="free", init=np.full(n, 10.0))
        )
        theta.update(dur.init_from_mean_sojourn(jnp.asarray(MEAN_SOJOURN_DAYS), nominal_dt()))
        if scenario.cov_duration:
            specs.append(
                ParamSpec(
                    name="duration.c",
                    shape=(n, 1),
                    domain="real",
                    mode="prior",
                    init=np.zeros((n, 1)),
                    prior=NormalPrior(sd=1.0),
                )
            )
            theta["duration.c"] = jnp.asarray(DURATION_COEF_STAR[:, None])
        engine_name: Literal["expanded", "explicit"] = "expanded"
    else:
        dur = GammaDuration(cov_idx=())
        specs.append(
            ParamSpec(name="duration.shape", shape=(n,), domain="positive", mode="free", init=np.full(n, 1.5))
        )
        specs.append(
            ParamSpec(
                name="duration.scale", shape=(n,), domain="positive", mode="free", init=np.full(n, 0.05)
            )
        )
        theta.update(dur.init_from_mean_sojourn(jnp.asarray(MEAN_SOJOURN_DAYS), nominal_dt()))
        engine_name = "explicit"

    # transition: baseline pi_rows, or covariate-conditioned a/b.
    layout = build_memory_layout(n, scenario.order)
    r_rows = layout.tuples.shape[0]
    if scenario.cov_transition:
        trans = JumpTransition(cov_idx=(0, 1))
        a = np.stack([_a_from_pi_row(PI_ROWS_TARGETS[c]) for c in layout.current_of])
        b = np.zeros((r_rows, n - 2, 2))
        b[:, :, 0] = TRANS_COEF_STAR[0]
        b[:, :, 1] = TRANS_COEF_STAR[1]
        specs.append(
            ParamSpec(name="transition.a", shape=a.shape, domain="real", mode="free", init=np.zeros_like(a))
        )
        specs.append(
            ParamSpec(
                name="transition.b",
                shape=b.shape,
                domain="real",
                mode="prior",
                init=np.zeros_like(b),
                prior=NormalPrior(sd=1.0),
            )
        )
        theta["transition.a"] = jnp.asarray(a)
        theta["transition.b"] = jnp.asarray(b)
    else:
        trans = JumpTransition()
        pi_rows = np.stack([PI_ROWS_TARGETS[c] for c in layout.current_of])
        specs.append(
            ParamSpec(
                name="transition.pi_rows",
                shape=pi_rows.shape,
                domain="simplex_ref",
                mode="prior",
                init=np.full(pi_rows.shape, 1.0 / (n - 1)),
                prior=DirichletPrior(alpha=1.5),
            )
        )
        theta["transition.pi_rows"] = jnp.asarray(pi_rows)

    spec = ModelSpec(
        n_states=n,
        n_series=1,
        emission=cast("EmissionFamily", emis),
        duration=cast("DurationFamily", dur),
        transition=cast("TransitionModel", trans),
        order=scenario.order,
        duration_conditioning="current",
        engine=engine_name,
        d_max=scenario.d_max,
        initial="uniform",
        param_specs=tuple(specs),
    )
    return spec, theta


SCENARIOS: dict[str, Scenario] = {
    "gbm_negbin": Scenario(name="gbm_negbin"),
    "transition_covariates": Scenario(name="transition_covariates", cov_transition=True),
    "duration_covariates": Scenario(name="duration_covariates", cov_duration=True),
    "order2": Scenario(name="order2", order=2, t_scale=2.5),
    # Mean sojourn is report-only here (owner decision): with both gamma shape and scale free the data
    # constrain it much less than in the NB scenarios (r fixed) -- at T=20000 the fit's likelihood exceeds the
    # truth's by ~12 nats (statistics, not the optimizer) and the 120-day regime misses 25% in 2 of 5 seeds
    # while an oracle on the same paths does not.
    "explicit_gamma": Scenario(
        name="explicit_gamma", duration="gamma", d_max=400, report_only=("mean_sojourn",)
    ),
    "fixed_drift": Scenario(name="fixed_drift", mu_fixed=True),
}


# --- simulation + fit + tolerance checks -------------------------------------------------------


def _space_and_engine(spec: ModelSpec, cfg: EstimationSection) -> tuple[StateSpace, InferenceEngine]:
    """Mirrors `estimation.map.fit`'s own space/engine construction (needed again for Hessian SE)."""
    if spec.engine == "explicit":
        return build_explicit(spec), ExplicitDurationEngine(chunk_size=cfg.chunk_size)
    return build_expanded(spec), ExpandedHMMEngine(parallel_scan=cfg.parallel_scan)


def derive_cov_seed(seed: int) -> int:
    """A plain int NumPy seed for the covariate generator, deterministic in `seed` alone."""
    return int(jax.random.bits(key_for(seed, _COV_PURPOSE), (), dtype=jnp.uint32))


def _empty_prepared(t_len: int, y: np.ndarray, dt: jnp.ndarray) -> PreparedDataset:
    """The "no transform applied" `PreparedDataset` wrapper (`inference.simulate.simulate_dataset`'s)."""
    stats = PrepStats(
        cov_mean=np.zeros(0), cov_std=np.ones(0), train_window=("", ""), cov_names=(), transforms={}, lags={}
    )
    return PreparedDataset(
        times=np.arange(t_len).astype("datetime64[D]").astype("datetime64[ns]"),
        observations=jnp.asarray(y),
        dt=dt,
        mask=jnp.ones(y.shape, dtype=bool),
        cov_transition=None,
        cov_duration=None,
        cov_emission=None,
        cov_names={"transition": (), "duration": (), "emission": ()},
        clock="trading",
        nominal_dt=nominal_dt(),
        prep_stats=stats,
        snapshot_ids=(),
    )


def _simulate_explicit(spec: ModelSpec, theta: Params, seed: int, t_len: int) -> PreparedDataset:
    """Segment-based simulation for the explicit engine.

    `inference.simulate.simulate_paths` steps via `space.transition_fn`, which `build_explicit`
    deliberately never attaches (`statespace/builders.py`: "the engine works on segments, never on
    `A_t`") -- found while adding this scenario. Draws regime segments directly instead: each
    segment's sojourn from `spec.duration`'s own law (NumPy, keyed off `seed` like the covariate
    generator), the next regime from `spec.transition.jump_probs_t` at the order-1 memory layout.
    """
    rng = np.random.default_rng(derive_cov_seed(seed) + 1)
    memory = build_memory_layout(spec.n_states, 1)
    pi = np.asarray(spec.transition.jump_probs_t(theta, None, memory))  # (N, N)
    shape = np.asarray(theta["duration.shape"])
    scale = np.asarray(theta["duration.scale"])

    regimes: list[int] = []
    current = int(rng.integers(spec.n_states))
    while len(regimes) < t_len:
        sojourn = rng.gamma(float(shape[current]), float(scale[current]))
        d = max(1, int(np.ceil(sojourn / nominal_dt())))
        regimes.extend([current] * d)
        current = int(rng.choice(spec.n_states, p=pi[current]))
    regime_arr = jnp.asarray(regimes[:t_len], dtype=jnp.int32)

    dt = jnp.full((t_len,), nominal_dt())
    y = np.asarray(spec.emission.sample(key_for(seed, _SIM_PURPOSE), regime_arr, dt, theta, None))
    return _empty_prepared(t_len, y, dt)


def _simulate(scenario: Scenario, spec: ModelSpec, theta: Params, seed: int, t_len: int) -> PreparedDataset:
    if spec.engine == "explicit":
        return _simulate_explicit(spec, theta, seed, t_len)

    space, _ = _space_and_engine(spec, EstimationSection())
    dt = jnp.full((t_len,), nominal_dt())
    z_pi = jnp.asarray(_ar1_covariates(t_len, 2, derive_cov_seed(seed))) if scenario.cov_transition else None
    z_d = jnp.asarray(_ar1_covariates(t_len, 1, derive_cov_seed(seed) + 1)) if scenario.cov_duration else None
    return simulate_dataset(
        key_for(seed, _SIM_PURPOSE),
        theta,
        spec,
        space,
        dt,
        z_pi,
        z_d,
        None,
        clock="trading",
        nominal_dt=nominal_dt(),
    )


@dataclass(frozen=True)
class ParamCheck:
    """One truth/estimate/tolerance comparison; part of a `RecoveryReport`."""

    name: str
    truth: float
    estimate: float
    se: float | None
    rule: str
    evaluated: bool
    passed: bool

    @property
    def category(self) -> str:
        """Tolerance category (concept Appendix D row): sigma, mu, mean_sojourn, jump or coefficient."""
        if self.name.startswith(("emission.scale", "emission.sigma")):
            return "sigma"
        if self.name.startswith(("emission.mean", "emission.mu")):
            return "mu"
        if self.name.startswith("mean_sojourn"):
            return "mean_sojourn"
        if self.name.startswith("pi["):
            return "jump"
        return "coefficient"

    def line(self) -> str:
        """One human-readable `truth/estimate/se/rule` line for an assertion message."""
        se_str = f"{self.se:.4g}" if self.se is not None else "n/a"
        return f"{self.name}: truth={self.truth:.4g} estimate={self.estimate:.4g} se={se_str} ({self.rule})"


@dataclass(frozen=True)
class RecoveryReport:
    """One scenario, one seed: truth/estimate/tolerance table + the underlying `FitResult`."""

    scenario: str
    seed: int
    t_len: int
    checks: tuple[ParamCheck, ...]
    fit: FitResult

    @property
    def passed(self) -> bool:
        """All *evaluated* checks pass (a check with no SE yet is excluded until the suite fills it in)."""
        return all(c.passed for c in self.checks if c.evaluated)

    def failures(self) -> str:
        """`; `-joined lines for every failed, evaluated check (empty message if none failed)."""
        lines = [c.line() for c in self.checks if c.evaluated and not c.passed]
        return "; ".join(lines) if lines else "no failing checks"


def _sigma_checks(theta_hat: Params) -> list[ParamCheck]:
    hat = np.asarray(theta_hat["emission.scale"])
    out = []
    for i in range(N_STATES):
        rel = abs(hat[i] - SIGMA_STAR[i]) / SIGMA_STAR[i]
        out.append(
            ParamCheck(
                f"emission.scale[{i}]",
                float(SIGMA_STAR[i]),
                float(hat[i]),
                None,
                f"rel <= {SIGMA_REL_TOL:.0%}",
                True,
                bool(rel <= SIGMA_REL_TOL),
            )
        )
    return out


def _mu_checks(theta_hat: Params, se: dict[str, np.ndarray] | None) -> list[ParamCheck]:
    hat = np.asarray(theta_hat["emission.mean"]).reshape(-1)
    se_mu = None if se is None else np.asarray(se.get("emission.mean")).reshape(-1)
    out = []
    for i in range(N_STATES):
        se_i = None if se_mu is None else float(se_mu[i])
        evaluated = se_i is not None
        passed = se_i is not None and abs(hat[i] - MU_STAR[i]) <= SE_MULT * se_i
        out.append(
            ParamCheck(
                f"emission.mean[{i}]",
                float(MU_STAR[i]),
                float(hat[i]),
                se_i,
                f"<= {SE_MULT:g} SE",
                evaluated,
                bool(passed),
            )
        )
    return out


def _mean_sojourn_checks(theta_hat: Params, spec: ModelSpec) -> list[ParamCheck]:
    hat_days = np.asarray(spec.duration.mean_sojourn(theta_hat, nominal_dt())) / nominal_dt()
    out = []
    for i in range(N_STATES):
        rel = abs(hat_days[i] - MEAN_SOJOURN_DAYS[i]) / MEAN_SOJOURN_DAYS[i]
        out.append(
            ParamCheck(
                f"mean_sojourn[{i}]",
                float(MEAN_SOJOURN_DAYS[i]),
                float(hat_days[i]),
                None,
                f"rel <= {MEAN_SOJOURN_REL_TOL:.0%}",
                True,
                bool(rel <= MEAN_SOJOURN_REL_TOL),
            )
        )
    return out


def _jump_checks(theta_hat: Params, spec: ModelSpec, scenario: Scenario) -> list[ParamCheck]:
    layout = build_memory_layout(N_STATES, scenario.order)
    z0 = jnp.zeros((2,)) if scenario.cov_transition else None
    pi_hat = np.asarray(spec.transition.jump_probs_t(theta_hat, z0, layout))  # (R, N)
    out = []
    for r, current in enumerate(layout.current_of):
        targets = [j for j in range(N_STATES) if j != current]
        truth_row = PI_ROWS_TARGETS[current]
        for k, target in enumerate(targets):
            truth = float(truth_row[k])
            hat = float(pi_hat[r, target])
            out.append(
                ParamCheck(
                    f"pi[{current}->{target}](r={r})",
                    truth,
                    hat,
                    None,
                    f"abs <= {JUMP_ABS_TOL:g}",
                    True,
                    bool(abs(hat - truth) <= JUMP_ABS_TOL),
                )
            )
    return out


def _coefficient_checks(
    theta_hat: Params, se: dict[str, np.ndarray] | None, scenario: Scenario
) -> list[ParamCheck]:
    out: list[ParamCheck] = []
    if scenario.cov_transition:
        hat = np.asarray(theta_hat["transition.b"])  # (R, N-2, 2)
        se_b = None if se is None else np.asarray(se.get("transition.b"))
        for k in range(2):
            truth = float(TRANS_COEF_STAR[k])
            hat_k = float(hat[0, 0, k])
            se_k = None if se_b is None else float(se_b[0, 0, k])
            evaluated = se_k is not None
            sign_ok = (hat_k > 0) == (truth > 0)
            passed = se_k is not None and sign_ok and abs(hat_k - truth) <= SE_MULT * se_k
            out.append(
                ParamCheck(
                    f"transition.b[{k}]",
                    truth,
                    hat_k,
                    se_k,
                    f"sign-correct, <= {SE_MULT:g} SE",
                    evaluated,
                    bool(passed),
                )
            )
    if scenario.cov_duration:
        hat = np.asarray(theta_hat["duration.c"]).reshape(-1)
        se_c = None if se is None else np.asarray(se.get("duration.c")).reshape(-1)
        for i in range(N_STATES):
            truth = float(DURATION_COEF_STAR[i])
            hat_i = float(hat[i])
            se_i = None if se_c is None else float(se_c[i])
            evaluated = se_i is not None
            sign_ok = (hat_i > 0) == (truth > 0)
            passed = se_i is not None and sign_ok and abs(hat_i - truth) <= SE_MULT * se_i
            out.append(
                ParamCheck(
                    f"duration.c[{i}]",
                    truth,
                    hat_i,
                    se_i,
                    f"sign-correct, <= {SE_MULT:g} SE",
                    evaluated,
                    bool(passed),
                )
            )
    return out


def run_scenario(
    name: str, seed: int, t_len: int | None = None, *, restarts: int = RESTARTS, device: str = "cpu"
) -> RecoveryReport:
    """Simulate `name`'s truth model at `seed`, MAP-fit it, and check every applicable tolerance.

    Hessian standard errors (`estimation.report.std_errors`) feed the `mu`/coefficient checks; when
    the Hessian is not positive-definite those checks come back `evaluated=False` (`run_recovery_suite`
    substitutes the 5-seed empirical spread instead, concept Appendix D's documented alternative).
    `restarts` defaults to the nightly suite's 4 but is overridable (E5-T05b's `selftest --quick`
    uses 2, per the ticket's "T=1500, restarts=2" smoke-test sizing).
    """
    scenario = SCENARIOS[name]
    t = t_len if t_len is not None else round(_default_t_len() * scenario.t_scale)
    spec, theta_star = _build_spec_and_truth(scenario)
    data = _simulate(scenario, spec, theta_star, seed, t)

    cfg = EstimationSection(
        restarts=restarts,
        init=["rolling_kmeans", "lattice_jitter", "random"],
        device=device,
    )
    result = fit(spec, data, cfg, seed)

    manager = ParamManager(spec.param_specs)
    space, engine = _space_and_engine(spec, cfg)
    objective = make_objective(spec, space, engine, manager, data, use_priors=cfg.use_priors)
    try:
        se = std_errors(objective, result.uparams, manager)
    except NotImplementedError:
        # `jax.hessian` needs `igamma`'s second derivative (w.r.t. the gamma shape parameter) when
        # `duration.shape` is free (scenario 5, explicit gamma durations); JAX has no JVP rule for
        # `igamma_grad_a`, so the *first*-order gradient (used by the fit itself) works but the
        # Hessian does not. `run_recovery_suite`'s empirical-spread fallback covers this the same way
        # it covers a non-positive-definite Hessian.
        se = None
    se_np = None if se is None else {k: np.asarray(v) for k, v in se.items()}

    checks: list[ParamCheck] = []
    checks += _sigma_checks(result.params)
    if not scenario.mu_fixed:
        checks += _mu_checks(result.params, se_np)
    checks += _mean_sojourn_checks(result.params, spec)
    checks += _jump_checks(result.params, spec, scenario)
    checks += _coefficient_checks(result.params, se_np, scenario)

    return RecoveryReport(scenario=name, seed=seed, t_len=t, checks=tuple(checks), fit=result)


def category_outcomes(reports: Sequence[RecoveryReport]) -> dict[str, tuple[int, int]]:
    """`{category: (seeds passing, seeds evaluated)}`.

    A seed passes a category when every *evaluated* check of that category passes in that seed; a seed
    without any evaluated check of the category (e.g. no `mu` check when `mu` is fixed) is not counted.
    """
    out: dict[str, tuple[int, int]] = {}
    for category in sorted({c.category for r in reports for c in r.checks if c.evaluated}):
        per_seed = [[c.passed for c in r.checks if c.evaluated and c.category == category] for r in reports]
        evaluated = [oks for oks in per_seed if oks]
        out[category] = (sum(1 for oks in evaluated if all(oks)), len(evaluated))
    return out


@dataclass(frozen=True)
class SuiteResult:
    """`run_scenario` over `seeds`, with the empirical 5-seed spread filling in missing SEs.

    Pass criterion (owner decision, E5-T05a AC1): every *gating* tolerance category (sigma, mu, mean
    sojourn, jump probabilities, covariate coefficients) must be met in at least 4 of the 5 seeds on its
    own; the categories need not be met in the same seeds. Categories in `report_only` are listed in
    `categories` and `summary()` but do not decide `passed`. `n_pass` (seeds meeting *every* tolerance at
    once) is kept as a stricter diagnostic.
    """

    scenario: str
    reports: tuple[RecoveryReport, ...]
    n_pass: int
    report_only: tuple[str, ...] = ()

    @property
    def categories(self) -> dict[str, tuple[int, int]]:
        """`{category: (seeds passing, seeds evaluated)}` for every category, see `category_outcomes`."""
        return category_outcomes(self.reports)

    def _failing_categories(self) -> dict[str, tuple[int, int]]:
        return {
            c: (ok, total)
            for c, (ok, total) in self.categories.items()
            if c not in self.report_only and ok < min(_MIN_PASS, total)
        }

    @property
    def passed(self) -> bool:
        """`True` iff every gating category passes in at least 4 of its seeds (all with fewer than 4)."""
        return not self._failing_categories()

    def failures(self) -> str:
        """Per failing gating category: its seed count and every failing check, for the assertion message."""
        lines = []
        for category, (ok, total) in self._failing_categories().items():
            bad = [
                f"seed {r.seed}: {c.line()}"
                for r in self.reports
                for c in r.checks
                if c.evaluated and c.category == category and not c.passed
            ]
            lines.append(
                f"{category}: {ok}/{total} seeds pass (need {min(_MIN_PASS, total)}) -- " + "; ".join(bad)
            )
        return "\n".join(lines) if lines else "no failing category"

    def summary(self) -> dict[str, object]:
        """A JSON-serializable record: outcome per category (gating or report-only) and every check."""
        return {
            "scenario": self.scenario,
            "t_len": self.reports[0].t_len if self.reports else None,
            "passed": self.passed,
            "n_seeds_all_checks": self.n_pass,
            "categories": {
                c: {"passing_seeds": ok, "evaluated_seeds": total, "report_only": c in self.report_only}
                for c, (ok, total) in self.categories.items()
            },
            "checks": [
                {
                    "seed": r.seed,
                    "name": c.name,
                    "category": c.category,
                    "truth": c.truth,
                    "estimate": c.estimate,
                    "se": c.se,
                    "rule": c.rule,
                    "passed": c.passed,
                }
                for r in self.reports
                for c in r.checks
                if c.evaluated
            ],
        }


def _apply_empirical_se(reports: Sequence[RecoveryReport]) -> tuple[RecoveryReport, ...]:
    """Re-evaluate any `evaluated=False` check using the across-seed spread of its own estimates."""
    by_name: dict[str, list[float]] = {}
    for report in reports:
        for check in report.checks:
            if not check.evaluated:
                by_name.setdefault(check.name, []).append(check.estimate)

    se_by_name = {
        name: float(np.std(np.asarray(values), ddof=1))
        for name, values in by_name.items()
        if len(values) >= 2
    }

    out = []
    for report in reports:
        new_checks = []
        for check in report.checks:
            if check.evaluated or check.name not in se_by_name:
                new_checks.append(check)
                continue
            se_emp = se_by_name[check.name]
            sign_ok = "sign-correct" not in check.rule or (check.estimate > 0) == (check.truth > 0)
            passed = sign_ok and abs(check.estimate - check.truth) <= SE_MULT * se_emp
            new_checks.append(replace(check, se=se_emp, evaluated=True, passed=bool(passed)))
        out.append(replace(report, checks=tuple(new_checks)))
    return tuple(out)


def run_recovery_suite(name: str, seeds: Sequence[int] = SEEDS, t_len: int | None = None) -> SuiteResult:
    """Run `name` over `seeds`; pass if every gating tolerance category holds in >= 4/5 seeds (Appendix D)."""
    reports = _apply_empirical_se([run_scenario(name, seed, t_len) for seed in seeds])
    n_pass = sum(1 for r in reports if r.passed)
    return SuiteResult(scenario=name, reports=reports, n_pass=n_pass, report_only=SCENARIOS[name].report_only)
