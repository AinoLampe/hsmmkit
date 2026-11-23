# E12-T06 — Label and boundary constraints as engine masks, `ConstraintError`, label-informed initialization, ADR-0019

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-SV01, FR-SV02, FR-SV03, D18 | M9 | Part II §14.10, §16.3 item 11, Appendix H (`data.labels`) |

## Goal
Express full or partial supervision as masks consumed by the unchanged engines, so every estimator works on labelled data, and initialize parameters from labels.

## Context (self-contained)
**Data**: `data.labels: {source: column | file, column | path, allowed_sets: bool, boundaries: {must_switch_at, must_not_switch_at}}`. Column labels: integer regime ids, NaN/`-1` = unlabelled → `labels (T, N)` bool with a single True per labelled row, all True when unlabelled. `allowed_sets: true` → the column holds sets encoded as a bitmask int or a list-valued column (pandas object) → `(T, N)` bool. File labels: parquet/CSV with `sequence_id, t, regime` (or `allowed` bitmask). Boundaries: step indices (per sequence when batched) into `PreparedDataset.boundaries` (Part I hook). `Preparer` fills `labels`/`boundaries`; `PaddedBatch.labels` True on padding (E12-T01a).
**Masks** (`engine/constraints.py`; dynamic arrays, Part II §16.3 item 11):
- Emissions: `ℓ_t(s) ← ℓ_t(s) + log L_t[regime_of[s]]` (`0` or `−inf`).
- Expanded `A_t`: `must_switch_at t` keeps only exit transitions `(m, i, r_i) → (m', j, 1)`, `j ≠ i` (all stay/advance entries `−inf`); `must_not_switch_at t` sets exit transitions to `−inf` (stay/advance kept). Rows are **not** renormalized (masks act on the joint likelihood).
- Explicit engine: `must_switch_at t` restricts `α_{t−1}` terms so that a segment ends at `t − 1` and a new one starts at `t`; `must_not_switch_at t` forbids segment boundaries between `t − 1` and `t`.
- Infeasibility: if `logsumexp α_T = −inf` → `ConstraintError(step=<first t with all α_t = −inf>, sequence=<id>)` raised outside jit (check returned arrays); e.g. NB minimum duration `r_i` incompatible with two boundaries closer than `r_i`.
- Masks are threaded through `loglik`, `forward_backward`, `viterbi`, the MAP objective, simulation is unaffected; decode on labelled data respects labels.
**Label initialization** (`init: labels`, registry kind `init`): emissions from labelled steps per regime (`m_step` on one-hot weights if the family has it, else moment matching documented per family), jump matrix from labelled segment transitions (row-normalized, Laplace-smoothed, topology-masked), durations from labelled segment lengths (mean → family `init_from_mean_sojourn`; right-censored last segments excluded), regimes without labels fall back to `random` for their parameters (warning).
**ADR-0019** (`docs/adr/0019-supervision-masks.md`, accepted).
Gate rows `data.labels`, `init: labels` removed.

## Creates
- `src/hsmmkit/engine/constraints.py`, label loading in `data/prepare.py`, `estimation/init.py` (`labels`), engine hooks, `ConstraintError` in `utils/errors.py`
- `tests/unit/test_e12_t06_constraints.py`, `tests/golden/test_e12_t06_masks.py`, `tests/integration/test_e12_t06_label_init.py`
- `docs/adr/0019-supervision-masks.md`, `docs/guide/core/supervision.md`

## Acceptance criteria
1. Fully labelled data reproduces the labels exactly under Viterbi (both engines).
2. Golden: masked engines equal a NumPy reference with the same masks ≤ 1e-8; unlabelled (`labels=None`) results bitwise equal to the unmasked engine.
3. Two `must_switch_at` boundaries 1 step apart with NB `r_i = 3` for every regime → `ConstraintError` naming the step and sequence.
4. `init: labels` recovers Gaussian means/scales and mean sojourns within 15 % on synthetic data with 30 % labelled steps, without optimization.
5. A compiled engine serves labelled and unlabelled sequences without recompiling (masks are dynamic).

## Tests
Unit, golden vs reference, integration, recompilation test.

## Out of scope
Semi-supervised weighting of labels (soft labels) — backlog.

## Connections
- Depends on: E12-T01b
- Blocks: E14-T06, E16-T01, E16-T05
- Relates to: E1-T01 (`labels`, `boundaries` fields), E14-T03 (supervised metrics)
