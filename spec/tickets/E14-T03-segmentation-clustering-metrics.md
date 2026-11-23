# E14-T03 — Segmentation and clustering metrics with Hungarian matching; ADR-0020

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EV01, FR-EV02 | M11 | Part II §14.13, §16.1 (Hungarian), §19 #12; Part I E7-T01 (`balanced_accuracy`) |

## Goal
Evaluate decoded segmentations against reference labels with boundary-based and partition-based metrics usable in CV, grids and reports.

## Context (self-contained)
All metrics registered (kind `metric`, `needs={"posterior", "labels"}`), operating per sequence on integer label arrays `q_true (T,)`, `q_pred (T,)` (Viterbi regimes), aggregated over sequences by length-weighted mean (documented):
- Boundaries `𝔅(q) = {t ≥ 1 : q_t ≠ q_{t−1}}`.
- `boundary_precision/recall/f1` at tolerance `τ = evaluation.boundary_tolerance`: one-to-one matching of predicted to true boundaries within `|t_p − t_t| ≤ τ` (greedy by distance, ties by earliest; equivalent to optimal matching on a line — prove in docstring); empty sets: precision = recall = 1 if both empty, else 0 for the empty side.
- `hausdorff`: `max(max_p min_t |p − t|, max_t min_p |p − t|)`; `nan` if one set is empty and the other not (documented).
- `covering`: `C = (1/T) Σ_{A ∈ segs(q_true)} |A| · max_{B ∈ segs(q_pred)} |A ∩ B| / |A ∪ B|`.
- Hungarian matching (`evaluation/metrics/matching.py`): confusion `C[i, k] = #{t: q_pred = i, q_true = k}`; maximize matched counts. **ADR-0020** decides between `scipy.optimize.linear_sum_assignment` as a runtime dependency and a vendored O(n³) implementation: measure `import scipy.optimize` time; if > 50 ms on the CI runner → vendor (Part II §19 #12). Either way the API is `hungarian(cost) -> (row_ind, col_ind)`.
- `hamming_matched` (accuracy after matching), `confusion_matched` (dict), `ari`, `nmi` (arithmetic-mean normalization, as scikit-learn default) — label-free, no matching.
- `balanced_accuracy` (Part I) switched from brute-force permutations to `hungarian` (results identical for `N ≤ 6`, now valid for any `N`).

## Creates
- `src/hsmmkit/evaluation/metrics/{segmentation,clustering,matching}.py`, `docs/adr/0020-hungarian.md`
- `tests/unit/test_e14_t03_segmentation.py`, `tests/unit/test_e14_t03_clustering.py`, dev-only comparison `tests/dev/test_e14_t03_vs_sklearn.py`

## Acceptance criteria
1. Closed-form cases exact: perfect segmentation (F1 = 1, Hausdorff = 0, covering = 1), all boundaries shifted by `τ` (F1 = 1) and by `τ + 1` (F1 = 0), single-segment vs many segments, empty boundary sets.
2. ARI/NMI vs `sklearn.metrics.adjusted_rand_score`/`normalized_mutual_info_score` ≤ 1e-12 on random labelings (dev-only).
3. `hungarian` equals brute-force optimum for random `N ≤ 6`; `balanced_accuracy` unchanged on Part I tests; invariant to label permutation.
4. ADR-0020 accepted with the measured import time; the import-time test (E10-T11) reflects the decision.

## Tests
Unit tests with closed-form fixtures, property tests (permutation invariance), dev-only sklearn comparison.

## Out of scope
Segmentation-aware training losses.

## Connections
- Depends on: E10-T08
- Blocks: E16-T05
- Relates to: E7-T01, E13-T06b (relabelling reuse), E14-T01
