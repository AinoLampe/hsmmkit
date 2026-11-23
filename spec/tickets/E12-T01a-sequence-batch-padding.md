# E12-T01a — `SequenceBatch`: splitting into sequences, padding to length buckets, `PaddedBatch`, ADR-0021

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-SQ01, FR-SQ03, FR-SQ04, D14 | M9 | Part II §14.3, §15.2 (`SequenceBatch`), §16.3 item 9; overview §5.11 |

## Goal
Represent many independent sequences of different lengths as single-sequence datasets plus a padded, bucketed array view that the batched engines (E12-T01b) consume.

## Context (self-contained)
```python
@dataclass(frozen=True)
class SequenceBatch:
    sequences: tuple[PreparedDataset, ...]      # B ≥ 1, each validated, same M, same covariate layout, same clock
    bucket_edges: tuple[int, ...]               # strictly increasing; T_max = smallest edge ≥ max T_b (else max T_b rounded up to a multiple of the last edge)
    ids: tuple[str, ...]                        # sequence ids (from sequence_column) in input order
    def padded(self) -> PaddedBatch: ...
    def unpad(self, arr: Array, *, time_axis: int = 1) -> tuple[Array, ...]: ...   # (B, T_max, …) → per-sequence (T_b, …)
    @property
    def lengths(self) -> np.ndarray: ...        # (B,)

@dataclass(frozen=True)
class PaddedBatch:                              # pytree, all leaves jnp arrays
    observations: Array   # (B, T_max, M)   padded with 0
    mask: Array           # (B, T_max, M)   False on padding
    valid: Array          # (B, T_max) bool True for t < T_b
    dt: Array             # (B, T_max)      padding dt = nominal_dt (positive dummy)
    cov_transition: Array; cov_duration: Array; cov_emission: Array    # (B, T_max, K_·), 0 on padding
    labels: Array | None  # (B, T_max, N) True on padding (all allowed)
    lengths: Array        # (B,) int32
    group_index: Array    # (B,) int32 initial-distribution group (FR-SQ02; 0 if none)
```
- **Splitting**: `Preparer.transform_batch(obs, covs) -> SequenceBatch` uses `ObservationSet.sequence_ids` (from `data.source.sequence_column`, E10-T04a): rows grouped by id preserving first-appearance order, then each group prepared as a single sequence with the **same** `PrepStats` (fit on the training rows of all sequences pooled). Lag ≥ 1 and own-lag generation (E11-T05) are applied **within** each sequence (no leakage across boundaries).
- **Group ids** from `data.sequences.group_by` (a column constant within a sequence; else `ConfigError`) → `PreparedDataset.group` and `group_index`.
- **Padding semantics** (Part II §14.3): engines must treat `valid == False` steps as `ℓ_t(s) = 0` and `A_t = I`; this ticket only provides `valid`; E12-T01b implements the engine side.
- **Buckets** bound recompilation: compiled shapes are `(B_bucket, T_bucket)` where `B` is padded up to the next power of two with empty sequences (`valid` all False, length 0 — they contribute log-likelihood 0) so that varying `B` does not recompile either.
- A single `PreparedDataset` is wrapped as `B = 1` by `SequenceBatch.of(ds)`.
- **ADR-0021** (`docs/adr/0021-multi-sequence.md`, accepted): `vmap` over per-sequence engines with length buckets; contracts stay single-sequence; no `B` dimension in `PreparedDataset` (Part II §19 #11).
- Gate rows `data.sequences`/`sequence_column` (data side) removed; fitting on `B > 1` remains gated until E12-T02.

## Creates
- `src/hsmmkit/data/batch.py`, `Preparer.transform_batch` in `data/prepare.py`
- `tests/unit/test_e12_t01a_batch.py`, `tests/property/test_e12_t01a_padding.py`
- `docs/adr/0021-multi-sequence.md`

## Acceptance criteria
1. `unpad(padded())` returns every sequence's arrays bitwise; `lengths` and `ids` preserved.
2. `T_max` equals the smallest bucket edge ≥ `max T_b`; `B` padded to a power of two with empty sequences.
3. Pooled `PrepStats`: standardization statistics equal those of the concatenated training rows; per-sequence lags never cross sequence boundaries (property test).
4. `group_by` column varying within a sequence → `ConfigError(path="data.sequences.group_by")`.
5. Loading the `core/segmentation_gaussian_ltr` preset fixture yields 50 sequences with the generator's lengths.

## Tests
Unit and property tests (padding invariance of the containers, bucket selection, lag isolation).

## Out of scope
Engine batching (E12-T01b); batch estimation (E12-T02).

## Connections
- Depends on: E10-T10
- Blocks: E12-T01b
- Relates to: E10-T04a (`sequence_ids`), E11-T05 (lags), E14-T01 (sequence folds)
