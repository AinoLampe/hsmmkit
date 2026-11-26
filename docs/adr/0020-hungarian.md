# ADR-0020: Vendored Hungarian algorithm instead of a scipy runtime dependency

Status: accepted (E14-T03).

## Context

Segmentation and clustering metrics (`hamming_matched`, `confusion_matched`, `balanced_accuracy`) need the optimal
assignment of predicted regime ids to reference ids. `scipy.optimize.linear_sum_assignment` solves it, but scipy
is a dev-only dependency of `hsmmkit` and would become a runtime dependency (Part II concept §19 #12: vendor the
algorithm if `import scipy.optimize` costs more than 50 ms).

## Measurement

`import numpy, scipy.optimize` against `import numpy` in a fresh interpreter (median of 7 runs, 4-core cloud VM,
scipy 1.15): **numpy 105 ms, with `scipy.optimize` 471 ms, i.e. +366 ms** (`-X importtime` reports ~390-450 ms
cumulative for `scipy.optimize`). That is far above the 50 ms threshold.

## Decision

- Vendor the shortest-augmenting-path Hungarian algorithm (`hsmmkit.evaluation.metrics.matching.hungarian`, `O(n^2 m)`,
  vectorized with NumPy) with the contract of `linear_sum_assignment` (minimum cost, rectangular, ascending
  `row_ind`). It is tested against the brute-force optimum for `N <= 6` and against scipy (dev-only) in the
  development comparison.
- scipy stays a dev dependency (reference values, tests); no runtime dependency is added. The import-time test
  (E10-T11) keeps `scipy` in its forbidden set: `import hsmmkit` and the metric modules never load it.
- `balanced_accuracy` (Part I) is switched from brute force over permutations to `hungarian`: identical values for
  `N <= 6`, valid for any number of regimes.

## Consequences

- No new runtime dependency and no import-time cost.
- A scipy bug fix or speed-up is not inherited; the vendored routine is small (about 50 lines) and covered by
  property tests.
- The boundary matching of the segmentation metrics is a separate, exact `O(K log K)` sweep on the line, not the
  Hungarian algorithm.
