# E11-T02 — Gaussian with `full` covariance (Cholesky parameterization)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EM01, FR-EM07 | M8 | Part II §14.2 (Multivariate Gaussian), Appendix F, Appendix G; Part I §4.6 (bijectors) |

## Goal
Extend `gaussian` to correlated multivariate observations with a Cholesky-parameterized covariance, closed-form sufficient statistics and a Normal-Wishart conjugate prior.

## Context (self-contained)
```
y_t | i ~ N( m_i(t) s(Δ_t),  Σ_i(t) s(Δ_t) ),   Σ_i = L_i L_iᵀ,  L_i lower-triangular, diag(L_i) = softplus(·)
Itô correction: m_i ← μ_i − ½ diag(Σ_i)
log-scale link (full): Σ_i(t) = D_t L_i L_iᵀ D_t with D_t = diag(exp(δ_iᵀ z^e_t))   (scales per series)
```
- New bijector `cholesky_lower` in `model.params`: `u ∈ ℝ^{M(M+1)/2}` → `L` with `softplus` on the diagonal; log-det-Jacobian `Σ_k log σ'(u_kk)` (softplus derivative); parameter spec `emission.chol (N, M(M+1)/2)` domain `cholesky`, `mean (N, M)`. `covariance: diag` keeps E10-T06 specs (`scale`) unchanged.
- Priors: `lognormal` on `diag(L)` (Appendix H), `lkj(eta)` optional on the correlation (log-density of LKJ via Cholesky factor, `η = 1` uniform) — LKJ is added to `model.dists` with scipy-free implementation, tested against a NumPy reference.
- `log_prob`: `−½[M log 2π + 2 Σ log diag(L̃) + ‖L̃⁻¹(y − μ̃)‖²]` with `L̃ = D_t L √s(Δ)`; triangular solve via `jax.scipy.linalg.solve_triangular`.
- `sufficient_stats(y, dt, gamma, z)` (no covariates, `scale_by_dt=False` or constant `Δ`): per state `n_i = Σγ`, `S1_i = Σγ y/s`, `S2_i = Σγ y yᵀ/s`; `m_step`: `μ = S1/n`, `Σ = S2/n − μμᵀ` (with `Δ`-scaling divide appropriately; ito-correction case → `None`, ECM fallback), Cholesky of `Σ + εI` (`ε = 1e-10`). Covariates present → `m_step` returns `None`.
- `conjugate_prior()`: Normal-Wishart `(m0, κ0, W0, ν0)` defaults weakly informative from data scale (documented); used by E13-T05a.
- Gate `model.emission.covariance: full` removed from `GATES`.

## Creates
- `src/hsmmkit/model/emission/gaussian.py` (full mode), `model/params.py` (`cholesky_lower` bijector), `model/dists.py` (`mvn_logpdf`, `lkj_logpdf`)
- `tests/unit/test_e11_t02_gaussian_full.py`, `tests/property/test_e11_t02_cholesky.py`; contract case `gaussian-full`

## Acceptance criteria
1. log-pdf vs `scipy.stats.multivariate_normal.logpdf` ≤ 1e-10 for random `M ∈ {2, 3, 5}`, with and without Δ-scaling.
2. Cholesky bijector log-det-Jacobian vs `jax.jacfwd` ≤ 1e-8; round trip ≤ 1e-12.
3. `m_step` equals the numerical optimum ≤ 1e-8 relative (contract check 7); `gaussian-full` passes `family_contract`.
4. `covariance: diag` outputs bitwise unchanged versus E10-T06 (golden).
5. Gate removed; Appendix H emission block builds (other gates may still fire).

## Tests
Unit/property tests above; contract suite; regression on diag.

## Out of scope
Multivariate Student-t (E11-T03); VB updates (E13-T05a uses the prior).

## Connections
- Depends on: E11-T01
- Blocks: E11-T03, E11-T05, E11-T06, E11-T08, E16-T02
- Relates to: E10-T06, E13-T02a, E13-T05a
