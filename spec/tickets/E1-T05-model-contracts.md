# E1-T05 — Model-side contracts: `ModelSpec`, `StateIndex`, `Posterior`, `PosteriorResult`, `FitResult`, `ForecastResult`, `BacktestResult`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-I01…I04, FR-X01 | M0 | §5.4 (model-side contracts), §4.7.1 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 5: `FitResult.estimator`). No behavioural change for the finance profile.

## Goal
Define the result and structure types shared by engines, estimation, inference and evaluation, with collapse helpers.

## Context (self-contained)
```python
class ModelSpec(eqx.Module):                # structure only, no data
    n_states: int; n_series: int
    emission: EmissionFamily; duration: DurationFamily; transition: TransitionModel
    order: int; duration_conditioning: Literal["current", "pair"]
    engine: Literal["expanded", "explicit"]; d_max: int
    initial: Literal["uniform", "stationary", "free"]
    param_specs: tuple[ParamSpec, ...]      # union of all component specs

class StateIndex(eqx.Module):               # static (hashable) index arrays
    regime_of: np.ndarray                   # (S,) int
    phase_of: np.ndarray                    # (S,) int, 1-based phase within regime
    memory_of: np.ndarray                   # (S, k-1) int; shape (S, 0) for order 1
    collapse: np.ndarray                    # C (S, N) 0/1, C[s, regime_of[s]] = 1
    n_expanded: int; n_states: int; order: int
    def memory_tuple_index(self) -> np.ndarray: ...   # (S,) index of the memory tuple (row of jump matrix) a state uses

@dataclass(frozen=True)
class Posterior:
    log_alpha: Array; log_beta: Array; smoothed: Array   # (T, S)
    loglik: float; index: StateIndex
    def regime_filtered(self) -> Array: ...  # softmax(log_alpha) @ C   (T, N)
    def regime_smoothed(self) -> Array: ...  # smoothed @ C
    def elapsed_posterior(self) -> Array: ...  # (T, N, D_max) explicit engine; phase marginals (T, N, max r) expanded

@dataclass(frozen=True)
class PosteriorResult:
    times: np.ndarray; filtered: Array; smoothed: Array          # (T, N)
    viterbi: np.ndarray; segments: list[tuple[int, int, int]]    # (start, end, regime), end inclusive
    augmented: Posterior; loglik: float; aic: float; bic: float; icl: float

@dataclass(frozen=True)
class FitResult:
    estimator: Literal["map"]    # Part II widens to "em" | "gem" | "vb" | "mcmc"; always "map" in 0.1.0
    params: Params; uparams: Array; objective: float; loglik: float
    restarts: pd.DataFrame       # columns: restart, init_strategy, objective, loglik, n_iter, converged, wall_s
    converged: bool; n_iter: int; std_errors: dict[str, Array] | None
    diagnostics: dict[str, Any]; spec: ModelSpec; prep_stats: PrepStats; config_resolved: dict

@dataclass(frozen=True)
class ForecastResult:
    origin_time: np.datetime64; horizon_times: np.ndarray   # (h,)
    regime_probs: Array        # (h, N)
    samples: Array             # (n_sim, h, M) per-step log-returns
    cum_quantiles: Array       # (h, Q, M)
    mean: Array; var: Array    # (h, M) of cumulative log-return
    quantile_levels: tuple[float, ...]; covariate_forecaster: str; seed: int

@dataclass(frozen=True)
class BacktestResult:
    folds: pd.DataFrame        # fold, train_start, train_end, test_start, test_end, fit_time
    metrics: pd.DataFrame      # fold × horizon × metric (long format: fold, horizon, metric, value)
    strategy: pd.DataFrame | None   # time, position, return, equity, turnover
    summary: pd.DataFrame      # aggregated per horizon/metric (mean, sd across folds)
```
`Posterior` is pytree-registered (arrays as leaves, `index` static). Emission/duration/transition Protocol types are imported under `TYPE_CHECKING` from `model/emission/base.py` etc.; this ticket creates those `base.py` files with the Protocol *signatures only* (bodies implemented in E3-T01/T03/T05) so `ModelSpec` type-checks now.

## Creates
- `src/regimekit/model/spec.py`, `src/regimekit/statespace/index.py` (StateIndex), `src/regimekit/engine/base.py` (Posterior + `InferenceEngine` Protocol), `src/regimekit/inference/results.py` (PosteriorResult, ForecastResult), `src/regimekit/estimation/results.py` (FitResult), `src/regimekit/evaluation/results.py` (BacktestResult), Protocol stubs in `model/emission/base.py`, `model/duration/base.py`, `model/transition.py`
- `tests/unit/test_e1_t05_contracts.py`

## Acceptance criteria
1. All dataclasses are frozen; `Posterior` and `StateIndex` usable as jit arguments/static closures.
2. `regime_filtered()`/`regime_smoothed()` verified on a hand-built 2-regime, 3-phase `StateIndex` (rows sum to 1, correct collapse).
3. `mypy --strict` passes with the Protocol stubs.

## Tests
Unit tests for collapse helpers and pytree behaviour.

## Out of scope
Building a `StateIndex` from a spec (E4-T02a); computing IC (E4-T05).

## Connections
- Depends on: E1-T03b
- Blocks: E4-T02a
- Relates to: E1-T01, all E4–E7 tickets
