# E11-T03 — Multivariate Student-t emission

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | FR-EM02 | M8 | Part II §14.2 (Multivariate Student-t), Appendix F |

## Goal
Generalize `student_t` to `M > 1` with full or diagonal scale matrix while keeping the univariate Part I family bitwise unchanged.

## Context (self-contained)
```
y_t | i ~ t_{ν_i}( m_i(t) s(Δ_t),  Σ_i(t) s(Δ_t) ),  ν_i > 2 (bijector 2 + softplus),  Σ_i = L_i L_iᵀ
log f = lgamma((ν+M)/2) − lgamma(ν/2) − (M/2) log(νπ) − log|L̃| − ((ν+M)/2) log(1 + q/ν),  q = ‖L̃⁻¹(y − μ̃)‖²
```
- Options as `gaussian` (`covariance: diag | full`, `scale_by_dt`, `ito_correction` not allowed → `ConfigError`, covariate links on mean and log-scale). `ν` modes `free | tied | fixed` via the Part I `ParamCfg` (`tie` groups).
- Param specs: `emission.mean (N, M)`, `emission.scale (N, M)` (diag) or `emission.chol` (full, E11-T02 bijector), `emission.nu (N,)`. For `M = 1, covariance: diag` the existing univariate code path and spec names (`m`, `s`, `nu` of Part I E3-T02) are kept — the multivariate path is a new code path selected when `M > 1` or `covariance: full`.
- `sufficient_stats`/`m_step` → `None` (ECM in E13-T02b); `conjugate_prior` → `None`.
- Sampler: `μ + L z / sqrt(g/ν)`, `z ~ N(0, I)`, `g ~ χ²_ν`.

## Creates
- `src/hsmmkit/model/emission/student_t.py` (multivariate path), `tests/unit/test_e11_t03_mv_t.py`; contract cases `student_t-mv-diag`, `student_t-mv-full`

## Acceptance criteria
1. log-pdf vs `scipy.stats.multivariate_t.logpdf` ≤ 1e-10 (`M ∈ {2, 3}`, diag and full).
2. `ν → ∞` (`ν = 1e8`) approaches the `gaussian` log-density within 1e-6.
3. Univariate Part I outputs bitwise unchanged (golden from E3-T02 tests).
4. Both new cases pass `family_contract`.

## Tests
Unit tests above, contract suite, regression.

## Out of scope
EM-specific M-steps for `ν` (ECM handles it).

## Connections
- Depends on: E11-T02
- Blocks: —
- Relates to: E3-T02, E13-T02b
