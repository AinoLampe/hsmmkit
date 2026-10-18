# E3-T01 — `EmissionFamily` protocol and GBM emission with covariate links and carry hooks

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M02, FR-M09, FR-M12, FR-X02; Part II FR-G03 | M1 | §4.2, §5.5 (`EmissionFamily`), §5.10 (carry); Part II §14.2, §15.3, §15.7 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 4: `support` and `scale_by_dt` on the protocol). No behavioural change for the finance profile.

## Goal
Implement the default emission: per-state GBM ⇒ Gaussian log-returns scaled by `Δ_t`, with linear drift and log-volatility covariate links.

## Context (self-contained)
```python
class EmissionFamily(Protocol):
    name: str; native_time: Literal["continuous", "discrete"]
    support: Literal["real", "count", "binary", "categorical"]   # checked against PreparedDataset.support at model build
    scale_by_dt: bool            # True: mean/variance scale with Δ_t (GBM); documented per family
    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]: ...
    def log_prob(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """y (T, M) observations (`PreparedDataset.observations`), dt (T,), z (T, K_e) or None → (T, N) log-densities per regime (summed over M)."""
    def sample(self, key: Array, regime: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """regime (T,) int, dt (T,) → (T, M)."""
    def init_carry(self) -> Any: ...          # None for GBM
    def step_carry(self, carry, y_t, theta) -> Any: ...   # identity for GBM
    def sufficient_stats(self, y, dt, gamma) -> Any | None: ...   # None in v1 (EM hook, E5-T06)
```
GBM in regime `i`: `y_t | i ~ N((μ_i(t) − σ_i(t)²/2) Δ_t, σ_i(t)² Δ_t)` with
```
μ_i(t)     = μ_i + γ_iᵀ z^e_t           (drift link; γ_i ∈ ℝ^{K_drift})
log σ_i(t) = log σ_i + δ_iᵀ z^e_t       (log-vol link; δ_i ∈ ℝ^{K_logvol})
```
`GBMFamily` declares `support = "real"`, `scale_by_dt = True`. Parameters are in clock units: **annualized** (`μ` per year, `σ` per √year) under `trading`/`calendar`, per step under `steps` (where `Δ_t ≡ 1`, so the density is simply `N(μ_i − σ_i²/2, σ_i²)`; no code branches on the clock). `param_specs` returns: `emission.mu (N, M)` domain real; `emission.sigma (N, M)` domain positive (default `constraint=increasing` when configured); `emission.gamma (N, K_drift)` and `emission.delta (N, K_logvol)` domain real with default `NormalPrior(0, coef_prior_sd)`, present only if the corresponding covariate list is non-empty. Config → specs mapping (`ParamCfg` for `mu`, `sigma`) is done here via `GBMFamily.from_config(EmissionCfg, n_states, n_series, cov_names)`. Masked observations are the engine's job (this function returns densities for all `t`).

Emission covariate columns: `z` is the `cov_emission` block; drift and log-vol may use different subsets → store column index arrays per link.

## Creates
- `src/regimekit/model/emission/base.py` (Protocol, replaces E1-T05 stub), `src/regimekit/model/emission/gbm.py`
- `tests/unit/test_e3_t01_gbm.py`, `tests/property/test_e3_t01_gbm.py`
- `docs/math/emissions.md`

## Acceptance criteria
1. `log_prob` vs a NumPy reference (`scipy.stats.norm.logpdf` with the same mean/variance) ≤ 1e-12 for random `(T=50, N=3, M=1)`, daily (`Δ=1/252`) and weekly (`Δ=1/52`) grids.
2. Appendix E numbers reproduced: calm `(0.08, 0.12)` daily mean/sd `0.029 % / 0.756 %`, turbulent `(−0.10, 0.35)` → `−0.064 % / 2.205 %` (4 significant digits).
3. Zero coefficients reproduce the covariate-free density; non-zero `δ` scales the variance by `exp(2 δᵀz)` (assert).
4. `sample` moments within 3 SE of analytic for `n=20000` per regime; deterministic per key.
5. `M > 1` shapes accepted (sum over series), jittable, differentiable in `theta`.
6. `GBMFamily.support == "real"` and `scale_by_dt is True`; with `dt = ones(T)` the log-density equals `scipy.stats.norm.logpdf(y, μ − σ²/2, σ)` ≤ 1e-12 (steps clock).

## Out of scope
Student-t (E3-T02); OU/jump/GARCH families.

## Connections
- Depends on: E1-T03b, E1-T04
- Blocks: E3-T02, E3-T06, E4-T03a, E6-T01, E9-T01, E9-T03
- Relates to: E5-T06 (`sufficient_stats`)
