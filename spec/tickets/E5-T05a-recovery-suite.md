# E5-T05a — Parameter-recovery suite (nightly)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-04, §1.4 criterion 1 | M3 | §6.4.4, Appendix D |

## Goal
Prove end-to-end estimation correctness: simulate from known `θ*`, fit, and assert recovery within the concept's tolerances for six scenarios.

## Context (self-contained)
Scenarios (`T=20000` by default, daily `Δ`, `N=3` unless noted, `restarts=4`, 5 seeds each; a scenario passes if every gating tolerance *category* (σ, μ, mean sojourn, jump probabilities, coefficients) is met in ≥ 4/5 seeds, each category counted on its own -- *amended per owner decision, see the amendment note below*):
1. GBM / NB `r=(3,3,2)`, order 1, no covariates.
2. + transition covariates (2 standardized AR(1) covariates, coefficients ±0.5).
3. + duration covariates (1 covariate, `c = (0.6, −0.6, 0.4)`; amended from `(0.3, −0.3, 0.2)`, which is as small as the standard error at `T=20000`).
4. Order 2 (`duration_conditioning: current`), no covariates; runs at 2.5× the default `T` (each memory row sees ~1/6 of the transitions).
5. Explicit gamma durations (`engine: explicit`, `D_max=400`); its mean-sojourn category is *report-only* (not part of the pass criterion).
6. Fixed-drift lattice (`mu` fixed at `(−0.30, 0.00, 0.20)`, σ free increasing).
Tolerances (Appendix D): σ ≤ 10 % rel; μ ≤ 2 SE (when free; SE from E5-T04 or from the 5-seed spread); mean sojourn ≤ 25 % rel; jump probabilities ≤ 0.15 abs; coefficients sign-correct and ≤ 2 SE.
Truth values: `σ* = (0.10, 0.18, 0.35)`, `μ* = (0.10, 0.00, −0.20)` (free in 1–5), mean sojourns `(60, 120, 30)` days, `π*` rows e.g. `(·, 0.7, 0.3), (0.5, ·, 0.5), (0.2, 0.8, ·)`. Use `constraint: increasing` on σ in all scenarios.
Harness: `tests/recovery/harness.py` with `run_scenario(name, seed) -> RecoveryReport` (table of truth/estimate/tolerance/pass) so E5-T05b can reuse it; reports written to `runs/recovery/<scenario>.json` (outcome per category, report-only ones included, plus every check) in CI artifacts.

**Amendment (owner decision, 2026-10-18).** At `T=5000` the truth gives only ~64 regime visits per path, so even an oracle that reads the true regime path meets the sojourn and jump tolerances together in only half of the seeds (and ≥ 4/5 seeds in 18 %). The tolerances are kept unchanged; the sample size is raised to `T=20000` (oracle: 99 %), order 2 to `T=50000`, the pass rule is applied per tolerance category (2-SE checks miss ~5 % by design), the duration-covariate truth is doubled, and the explicit-gamma sojourn category is report-only (free gamma shape and scale make the mean weakly identified: the fit's likelihood beats the truth's by ~12 nats while an oracle passes). Evidence and results: `docs/guide/validation.md`, `spec/STATUS.md`. Concept §6.4.4 / Appendix D still say `T=5000` and are not edited here.

## Creates
- `tests/recovery/{harness,scenarios,test_e5_t05a_recovery}.py` (`@pytest.mark.slow`)
- `docs/guide/validation.md` (what the suite checks)

## Acceptance criteria
1. All six scenarios pass in the full (weekly) job: every gating tolerance category holds in ≥ 4/5 seeds (explicit gamma's mean-sojourn category is reported only).
2. Each failed tolerance is reported with truth/estimate/SE in the assertion message.
3. The nightly job (`nox -s recovery_smoke`: one scenario at `T=5000`, asserting the pipeline and the σ category) finishes in ≤ 60 min on the CI runner; the full suite runs weekly (`nox -s recovery`, `T=20000` by default, sizes adjustable via the `REGIMEKIT_RECOVERY_T` env var).

## Out of scope
`selftest` CLI (E5-T05b).

## Connections
- Depends on: E5-T03, E6-T01, E4-T06b (scenario 5), E5-T04 (SE)
- Blocks: E5-T05b, E8-T04 (release criterion)
- Relates to: E0-T03 nightly job
