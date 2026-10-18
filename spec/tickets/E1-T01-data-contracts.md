# E1-T01 — Data contracts: `ObservationSet`, `CovariateSeries`, `PrepStats`, `PreparedDataset`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D07, FR-M12, D10; Part II FR-G01, FR-G03 | M0 | §5.4 (data contracts), §4.1; Part II §15.2, §15.7 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendments 1–3: `observations`, optional supervision fields, three clocks, profile import rule). No behavioural change for the finance profile.

## Goal
Define the frozen, validated, pytree-registered dataclasses that are the only shared types between data layer and model layer.

## Context (self-contained)
```python
@dataclass(frozen=True)
class ObservationSet:            # raw per provider, before alignment
    symbols: tuple[str, ...]     # M
    times: np.ndarray            # (T0,) datetime64[ns] UTC, strictly increasing
    prices: np.ndarray           # (T0, M) float64, adjusted close (or configured field)
    interval: str                # "1h" | "1d" | "1wk" | "1mo"
    adjusted: bool; provider: str
    snapshot_id: str             # sha256(provider, symbols, interval, adjusted, raw parquet bytes)
    meta: dict[str, Any]

@dataclass(frozen=True)
class CovariateSeries:
    name: str; times: np.ndarray; values: np.ndarray   # (T1,) float64, NaN allowed
    provider: str; series_id: str; snapshot_id: str; frequency: str

@dataclass(frozen=True)
class PrepStats:                 # fitted on the training window only
    cov_mean: np.ndarray; cov_std: np.ndarray          # (K,)
    train_window: tuple[str, str]; cov_names: tuple[str, ...]
    transforms: dict[str, list[str]]; lags: dict[str, int]

@dataclass(frozen=True)
class BoundaryConstraints:       # Part II supervision hook; always None in 0.1.0
    must_switch_at: tuple[int, ...] = ()
    must_not_switch_at: tuple[int, ...] = ()

@dataclass(frozen=True)
class PreparedDataset:           # the ONLY object the model layer consumes
    times: np.ndarray            # (T,) grid end times
    observations: Array          # (T, M) float64 observations y_t (log-returns in the finance profile)
    dt: Array                    # (T,) Δ_t in years, > 0
    mask: Array                  # (T, M) bool, True = observed
    cov_transition: Array | None # (T, K_pi) already lagged & standardized
    cov_duration: Array | None   # (T, K_d)
    cov_emission: Array | None   # (T, K_e)
    cov_names: dict[str, tuple[str, ...]]   # keys: "transition","duration","emission"
    clock: Literal["trading", "calendar", "steps"]; nominal_dt: float   # median(dt); 1.0 under "steps"
    prep_stats: PrepStats; snapshot_ids: tuple[str, ...]
    # Part II hooks — fixed defaults in 0.1.0; declared now so the field set never changes again
    support: Literal["real", "count", "binary", "categorical"] = "real"
    n_categories: int | None = None
    labels: Array | None = None             # (T, N) bool allowed-set per step; None = unlabelled
    boundaries: BoundaryConstraints | None = None
    group: str | None = None                # initial-distribution group id
    def window(self, start: int, end: int) -> "PreparedDataset": ...   # slices T, keeps everything else
    def validate(self) -> None: ...
```
`validate()` raises `DataError(path=<field>)` on wrong ndim/dtype/length, non-monotonic times, `dt <= 0`, non-finite observations where `mask` is True, covariate rows ≠ T, `cov_names` keys/lengths mismatching the arrays, `support == "real"` with a non-float dtype (the other supports only check dtype/ndim in 0.1.0: `count` non-negative integer, `binary` in `{0,1}`, `categorical` integer in `[0, n_categories)` with `n_categories` required), `labels` present but not `(T, N)` bool, `boundaries` indices outside `[1, T−1]`, and `clock == "steps"` with any `dt != 1`. There is **no** `returns` alias anywhere in the core (Part II FR-G01); all v1 code uses `observations`. Pytree registration: array fields (including `labels`, which may be `None`) are leaves; strings/dicts/`PrepStats`/`support`/`boundaries`/`group` are aux data (use `jax.tree_util.register_dataclass` with `data_fields`/`meta_fields`). `nominal_dt` recomputed by `window()`? No — kept from the parent (it is a property of the grid, not the slice).

## Creates
- `src/regimekit/data/contracts.py`
- `tests/unit/test_e1_t01_contracts.py`, `tests/unit/test_e1_t01_layer_rule.py`
- `docs/guide/contracts.md` (short)

## Acceptance criteria
1. Each `validate()` rule has a failing test with a typed `DataError` carrying the field path.
2. `window(a, b)` preserves `dt`, `mask`, all covariate blocks, `labels` (sliced), `cov_names`, `prep_stats`, `snapshot_ids`, `support`, `group`, and `window(0, T)` equals the original; `boundaries` are re-indexed to the window and entries outside it dropped.
3. `jax.tree_util.tree_leaves(dataset)` returns only the array fields; `jax.jit` of a function taking a `PreparedDataset` works.
4. Layer rule test: scanning `src/regimekit/{model,statespace,engine,estimation,inference}` finds no import of `regimekit.data.*` other than `regimekit.data.contracts` (AST-based; runs in the default suite).
5. Profile rule test (same AST scanner, overview §4.3): no module under `src/regimekit/{config,data,model,statespace,engine,estimation,inference,evaluation,persistence}` imports `regimekit.finance`; `cli/` and `project.py` are exempt.
6. A dataset built with the defaults (`support="real"`, `labels=None`, `boundaries=None`, `group=None`, `clock="trading"`) validates and behaves exactly as before the amendment; a `clock="steps"` dataset with `dt = ones(T)` validates, with `dt = 1/252` it fails on path `dt`.

## Tests
Unit tests for validation, windowing, pytree behaviour; AST layer-rule test.

## Out of scope
Building datasets from providers (E2-T06/T07); model-side contracts (E1-T05).

## Connections
- Depends on: E0-T05
- Blocks: E1-T02a, E2-T01, E2-T06, E6-T03
- Relates to: E1-T05, E9-T01 (`M > 1`)
