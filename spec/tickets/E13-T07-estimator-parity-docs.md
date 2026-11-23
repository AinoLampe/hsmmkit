# E13-T07 — Estimator parity suite, estimator decision guide, `compare --estimators`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES06, FR-EV04, NFR-13, G10 | M10 | Part II §12.3 (0.3.0 criteria), §16.6 (`estimators.md`), Appendix G |

## Goal
Prove and document how the five estimators relate, and let users compare them on their own data in one command.

## Context (self-contained)
- **Parity suite** (`tests/parity/`, nightly `-m slow` + a fast subset on PRs): consolidates the rows written by E13-T02b (`em_map`), E13-T03b (`gem_map`), E13-T05b (`vb_map`), E13-T06a/b (`mcmc_truth`, `gibbs_nuts`) into one generated table `docs/generated/estimator_parity.md` with columns: model, estimators compared, check, tolerance (Appendix G), result, seeds passed. CI job `parity` regenerates it and fails if any row regresses.
- **`FitResult` uniformity** (FR-ES06): a contract test asserts for every estimator: `estimator` field, finite `loglik`, `params` on the constrained scale, `uparams` consistent with `params` (`manager.flatten`), `elbo` set iff VB, `posterior_samples` set iff MCMC, `pruned_states` tuple; bundles round-trip for every estimator (E8-T01 rules; MCMC samples saved as `posterior.nc` in `extensions/hsmmkit-mcmc/` only when `persistence.save_samples: true`).
- **`compare --estimators map,em,vb`**: runs each estimator on the same config/data/seed and writes `compare.parquet`/`compare.md` with columns `estimator, loglik, objective or elbo, n_params, bic, icl, wall_s, n_iter, converged, max |Δθ| vs MAP`, plus CV scores if `evaluation.cv` is set (from E14-T01 when available; column omitted otherwise).
- **`docs/guide/estimators.md`**: decision table (MAP / EM / GEM / VB / MCMC) × (requirements: engine, conjugacy, covariates; cost per iteration in `O(·)`; memory; outputs; failure modes: local optima, label switching, ELBO looseness, divergences; when to use), worked example on the Appendix E fixture, links to the parity table.

## Creates
- `tests/parity/conftest.py`, `scripts/gen_parity_table.py`, `docs/generated/estimator_parity.md`, `.github/workflows/nightly.yml` (`parity` job)
- `estimation/compare.py` or `evaluation/compare.py` extension (`--estimators`), `tests/integration/test_e13_t07_compare_estimators.py`, `tests/contract/test_e13_t07_fitresult.py`
- `docs/guide/estimators.md`

## Acceptance criteria
1. Parity table generated in CI with every row passing; a deliberately broken M-step (test fixture) makes the corresponding row fail.
2. `FitResult` contract passes for all five estimators; bundle round-trip bitwise for MAP/EM/GEM/VB (params) and for MCMC (posterior means; samples when enabled).
3. `hsmmkit compare CONFIG --estimators map,em,vb` produces the documented table on the Appendix E fixture.
4. `estimators.md` lists assumptions, cost and failure modes for each estimator and is linked from the docs nav.

## Tests
Parity suite, contract test, CLI integration.

## Out of scope
Grid over estimators × structures (E14-T02).

## Connections
- Depends on: E13-T02b, E13-T03b, E13-T05b, E13-T06b
- Blocks: E14-T02, E17-T01, E17-T03
- Relates to: E7-T04 (`compare`)
