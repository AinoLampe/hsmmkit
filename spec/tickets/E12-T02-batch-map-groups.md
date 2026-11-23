# E12-T02 — Batch estimation (MAP), per-group initial distributions, batched simulate

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-SQ01, FR-SQ02, NFR-12 | M9 | Part II §14.3, §12.3 (0.2.0 criteria); Part I §4.8, E5-T01…T03, E6-T01 |

## Goal
Fit, score and simulate collections of sequences with the MAP estimator, with an initial distribution shared or free per sequence group.

## Context (self-contained)
- **Objective** (`estimation/objective.py`): accepts `PreparedDataset | SequenceBatch`; `J(u) = −Σ_b ℓ_b(θ(u)) − log p(θ) − log|det J|` using `batched_loglik` (E12-T01b). `value_and_grad` stays the single JIT boundary; data are dynamic arguments (no recompilation across folds/batches of the same bucket).
- **Initial distribution groups** (FR-SQ02): config `model.initial_state` (`uniform | stationary | free`) plus `model.initial_groups: "shared" | "per_group" = "shared"`. `per_group` with `free` creates spec `initial.logits (G, S_phase1)` (G = number of distinct groups in the training batch, ordered by first appearance; stored in `FitResult.diagnostics["initial_groups"]` and the bundle so prediction maps group names → rows; unknown group at prediction → shared fallback row = mean of group logits, with a warning).
- **Init strategies on batches**: `rolling_kmeans` pools rolling-scale features over sequences (windows never cross boundaries); `lattice_jitter`, `random` unchanged; run lengths for `λ` init computed per sequence and pooled.
- **Multi-start/restarts** unchanged (E5-T03); determinism per seed.
- **Fit report**: occupancy, durations, restart dispersion computed over the batch; per-sequence loglik table `diagnostics["per_sequence"]` (id, T, loglik).
- **Score**: `Project.score(fit, data)` → total and per-sequence loglik.
- **Simulate**: `simulate(params, spec, lengths: Sequence[int], key, groups=None)` → `SequenceBatch` (vmapped E6-T01 simulator, keys `key_for(seed, b)`), used for synthetic batch fixtures and the recovery test below.
- Gate rows for fitting `B > 1` and for `model.initial_groups: per_group` removed (`data.sequences` fully available).
- Benchmark: one MAP objective+gradient evaluation at NFR-12 size recorded (no hard budget; EM budget is E13-T02a).

## Creates
- edits: `estimation/{objective,map,init,report}.py`, `inference/simulate.py`, `project.py` (`score`), `statespace/builders.py` (initial groups)
- `tests/integration/test_e12_t02_batch_fit.py`, `tests/recovery/test_e12_t02_batch_recovery.py` (`-m slow`), `tests/benchmark/test_e12_t02_bench.py`

## Acceptance criteria
1. Recovery on 50 sequences × `T = 200` (Gaussian diag, 3 states, NB `r = 2`) matches single-sequence recovery on `T = 10 000` within Appendix D/G tolerances (σ ≤ 10 %, sojourn ≤ 25 %, π ≤ 0.15; ≥ 4/5 seeds).
2. `per_group` initial distributions: two groups with different true initial regimes are recovered (correct argmax per group) on synthetic data.
3. A batch with `B = 1` reproduces the single-sequence fit bitwise (`uparams`).
4. Batched simulate is deterministic per key and the empirical sojourn mean matches the analytic within 3 SE.
5. Benchmark recorded; `core/segmentation_gaussian_ltr`'s data part loads and a MAP fit (with gated features disabled in a test variant) runs on the 50-sequence fixture.

## Tests
Integration, slow recovery, determinism, benchmark.

## Out of scope
EM/GEM/VB/MCMC (E13); labels (E12-T06).

## Connections
- Depends on: E12-T01b
- Blocks: E13-T06a, E14-T01, E14-T04, E16-T01
- Relates to: E5-T03, E6-T01
