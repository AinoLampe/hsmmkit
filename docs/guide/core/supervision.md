# Supervision: labels and boundaries

`data.labels` gives full or partial supervision; the engines treat it as a mask on the state paths (ADR-0019).

```yaml
data:
  labels:
    source: column          # or: file
    column: activity        # regime ids 0..N-1; NaN or -1 = unlabelled
    allowed_sets: false     # true: a bitmask int, or a list-valued column, of allowed regimes
    boundaries:
      must_switch_at: [120]       # a regime change between steps 119 and 120
      must_not_switch_at: [40, 41] # no change between steps 39/40 and 40/41
estimation:
  init: [labels, random]
```

* **Column labels** come from a column of the data source. With `source: file` the file (parquet or CSV) has the
  columns `sequence_id` (optional for one sequence), `t` (0-based row within the sequence) and `regime` or
  `allowed` (bitmask).
* **Boundaries** are step indices into each sequence (`1 <= t <= T-1`); indices beyond a shorter sequence are
  ignored for it.
* A likelihood of zero (for example two must-switch steps closer than the minimum duration of a negative-binomial
  regime) raises `ConstraintError` with the `step` and, for a batch, the `sequence`; the CLI exit code is 3.
* `init: labels` starts the optimizer from the labelled steps: per-regime means and scales, label-change counts for
  the jump matrix and the mean length of complete label runs for the durations. Regimes without labels use the
  `random` init (with a warning).
* Constrained likelihoods are likelihoods of the data *and* the constraints: do not compare them with unconstrained
  fits.
