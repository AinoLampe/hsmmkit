# E13-T02a — `EMEstimator` (expanded engine): closed-form M-steps, convergence, ADR-0018

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES01, FR-ES06, NFR-13, D17 | M10 | Part II §14.6 (Baum-Welch with pooling), §15.3 (`Estimator`); Part I §4.8, E5-T06 |

## Goal
Provide Baum-Welch EM for the expanded engine as a peer estimator of MAP for models whose components have closed-form M-steps, superseding the GBM-only EM of E5-T06.

## Context (self-contained)
```python
class EMEstimator:                     # estimation/em.py, registered estimator "em"
    name = "em"
    def fit(self, spec, data: PreparedDataset | SequenceBatch, cfg: EstimationSection, key) -> FitResult: ...
```
Iteration `k`: E-step `ExpandedEStats` (E13-T01a) at `θ_k`; M-step:
- **Emissions**: `stats = family.sufficient_stats(y, dt, gamma_regime, z)`; `θ_e ← family.m_step(stats)`; if `use_priors` and the family provides `map_step(stats, prior)` use it (MAP-EM), else ML. Families returning `None` → handled in E13-T02b (ECM); in this ticket such configs raise `ConfigError("component <x> has no closed-form M-step; available from E13-T02b")`.
- **Phase advance** (constant `Δ`, no duration covariates): `a_k = advance_k / occupancy_k` per rate row; `λ_k = −log(1 − a_k) / nominal_dt` (geometric/NB families expose `from_advance_prob`). Irregular `Δ` or duration covariates → ECM (E13-T02b).
- **Jump matrix** (no transition covariates): `π_rj ∝ jump[r, j] + (α_dir − 1 if use_priors)` over allowed targets (topology), converted to the masked-logit parameterization; transition covariates → ECM.
- **Initial distribution**: `free` → `ρ_g ∝ gamma1[g]`; `uniform`/`stationary` unchanged.
- Parameters in `fixed` mode are kept; `tied` parameters: sufficient statistics summed over the tie group before the M-step (generic helper); `bounded`: closed-form result clipped to bounds, with a warning in the fit report.
Convergence (Part II §14.6): stop when `Δℓ ≤ tol_f · max(1, |ℓ|)` or `max_iter`; assert monotonicity in debug mode (`HSMMKIT_DEBUG=1`): violation > 1e-10 raises `NumericalError`. Multi-start: `cfg.restarts` starts from the init strategies (same keys as MAP); best final `ℓ` (or log-posterior) wins. `FitResult(estimator="em", loglik, objective=−ℓ − log p, n_iter, converged, restarts table with ℓ trace)`, `uparams` from `manager.flatten(θ)` so bundles/decode work unchanged.
**E5-T06**: `estimation/em.py` of Part I (GBM-only EM) is replaced; its tests are kept as regression (GBM options of `gaussian`, no covariates).
**ADR-0018** (`docs/adr/0018-estimator-peers.md`, accepted): EM/GEM/VB/MCMC as peers behind `Estimator`, MAP reference and parity oracle; capability discovery via optional family methods.
Gate `estimation.method: em` removed.

## Creates
- `src/hsmmkit/estimation/em.py` (rewritten), `estimation/mstep.py` (generic tie/bound helpers, transition/duration M-steps), `Estimator` registry kind `estimator` (add to `plugins.Kind`)
- `tests/integration/test_e13_t02a_em.py`, `tests/property/test_e13_t02a_monotone.py`
- `docs/adr/0018-estimator-peers.md`

## Acceptance criteria
1. `ℓ` non-decreasing on 3 synthetic datasets each for Gaussian (diag/full), Poisson, Categorical (violations ≤ 1e-10).
2. Parity with MAP (`use_priors=false`, same start) on Gaussian/Poisson/Categorical with geometric and NB durations: `θ` rel ≤ 1e-4, loglik rel ≤ 1e-6 (Appendix G).
3. E5-T06 regression tests pass with the new implementation.
4. Batched data (E12) and topology masks are supported; constraints (labels) respected.
5. NFR-12: one EM iteration at `B = 64, T = 1000, N = 4, S = 12` ≤ 400 ms (benchmark recorded).

## Tests
Integration (parity), property (monotonicity), regression, benchmark.

## Out of scope
ECM, tempering (E13-T02b); explicit engine (E13-T03a/b).

## Connections
- Depends on: E13-T01a, E11-T04
- Blocks: E13-T02b, E13-T03b
- Relates to: E5-T06, E11-T02, E11-T06 (responsibility M-steps)
