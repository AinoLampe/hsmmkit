# E10-T06 — Core `gaussian` emission (diagonal, `M ≥ 1`, `scale_by_dt`, `ito_correction`); `gbm` as a finance alias

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-EM01 (diag part), FR-EM06, FR-G01, FR-G09 | M7a | Part II §14.2 (Multivariate Gaussian), Appendix F, §19 #10; Part I §4.2, E3-T01, E3-T02 |

## Goal
Give the core a neutral Gaussian family that reproduces GBM exactly when configured so, re-express all core tests with it, and keep `gbm` as a finance alias with unchanged parameter names and numbers.

## Context (self-contained)
`model/emission/gaussian.py`, registered as `gaussian`:
```
y_t | i ~ N( m_i(t) s(Δ_t),  diag(σ_i(t)²) s(Δ_t) ),   s(Δ) = Δ if scale_by_dt else 1
m_i(t)      = μ_i + γ_iᵀ z^e_t − (½ σ_i(t)² if ito_correction else 0)
log σ_i(t)  = log σ_i + δ_iᵀ z^e_t
```
- `support = "real"`, `native_time = "continuous" if scale_by_dt else "discrete"`, options `scale_by_dt: bool = False`, `ito_correction: bool = False`, `covariance: Literal["diag"] = "diag"` (`"full"` is gated until E11-T02).
- Param specs: `emission.mean (N, M)` real, `emission.scale (N, M)` positive (supports `constraint: increasing` per series), `emission.gamma (N, K_mean)`, `emission.delta (N, K_logscale)` with `NormalPrior(0, coef_prior_sd)` when covariates are configured. Config covariate keys: `covariates: {mean: [...], log_scale: [...]}`.
- **Operation order must match E3-T01** so that GBM options give bitwise-identical log-densities: compute `σ_t = σ_i · exp(δᵀz)`, `var = σ_t² · Δ`, `mean = (μ_t − ½σ_t²) · Δ`, `log_prob = −½(log(2π var) + (y − mean)² / var)` summed over `M` — reuse the E3-T01 helper rather than rewriting it.
- `sample`, `init_carry`/`step_carry` (identity), `sufficient_stats` (weighted sums `Σw, Σw y/s, Σw y²/s` per state and series; used by E13).
**GBM alias** (`finance/emission/gbm.py`): `GBMFamily(GaussianFamily)` fixed to `scale_by_dt=True, ito_correction=True`, registered as `gbm` (provenance finance) with its **own** spec names `emission.mu`, `emission.sigma`, config keys `mu`, `sigma`, covariate keys `drift`, `log_vol` (Part I names) mapped onto the parent's arrays — so v1/v2 finance configs and 0.1.x/0.2.x bundles keep working without renames.
**MS-GARCH spike** (E9-T03, `src/regimekit/model/emission/msgarch.py`, experimental, not in the family registry): move it unchanged to `src/regimekit/finance/emission/msgarch.py` together with GBM (it is finance-specific, Part II §15.8) and move its spike tests under the `finance` marker; the carry hooks (`init_carry`/`step_carry`) and the engine branch stay in the core.
**Student-t** stays core (`student_t`); its `scale_by_dt` default becomes `False` in the core; the v1→v2 finance migration (E10-T02b) already sets `scale_by_dt: true` for finance configs.
**Core tests** (golden vs NumPy reference, recovery scenarios, selftest, benchmarks, E9-T01 product tests): replace `family: gbm` by `family: gaussian, scale_by_dt: true, ito_correction: true` and `mu/sigma` by `mean/scale` (the reference implementation `tests/reference/ref_gbm.py` is renamed `ref_gaussian.py` with the same math). Finance keeps its own GBM tests (marker `finance`).

## Creates
- `src/regimekit/model/emission/gaussian.py`, `src/regimekit/finance/emission/gbm.py` (moved from `model/emission/gbm.py`), `src/regimekit/finance/emission/msgarch.py` (moved from `model/emission/msgarch.py`)
- `tests/unit/test_e10_t06_gaussian.py`, `tests/finance/test_e10_t06_gbm_alias.py`; edits across core test modules
- `docs/math/emissions.md` (Gaussian section; GBM paragraph moves to finance docs)

## Acceptance criteria
1. `gaussian(scale_by_dt=True, ito_correction=True)` log-density equals the v0.2.0 GBM log-density bitwise on daily/weekly/steps fixtures (`T=50, N=3, M ∈ {1,2}`); if bitwise is impossible on a backend, ≤ 1e-14 absolute and documented.
2. `gaussian` with defaults equals `scipy.stats.norm.logpdf(y, μ, σ)` ≤ 1e-12.
3. Appendix A v1 config (family `gbm`, `mu` fixed, `sigma` increasing) builds the same `ParamManager` layout and the v0.2.0 golden values are bitwise; 0.1.x/0.2.x bundle parameter names load.
4. With `HSMM_CORE_ONLY=1` the full core suite (golden, property, recovery smoke, selftest) passes using `gaussian` only.
5. Allow-list entries tagged `E10-T06` gone.

## Tests
Gaussian unit/property tests (moments, Δ-scaling, Itô term, covariate links), alias parameter-name mapping, golden test.

## Out of scope
Full covariance (E11-T02); multivariate Student-t (E11-T03); `family_contract` (E11-T01).

## Connections
- Depends on: E10-T02b
- Blocks: E10-T07b
- Relates to: E3-T01, E3-T02, E9-T01, E11-T02
