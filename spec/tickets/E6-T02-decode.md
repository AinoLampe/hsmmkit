# E6-T02 — `decode` API and CLI: filtered/smoothed/Viterbi/segments, augmented export, `posterior.parquet`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-I01, FR-I02, FR-C02, FR-P02 | M4 | §5.4 (`PosteriorResult`), §5.8 Results, §5.9 |

## Goal
Turn a fitted model plus data into a `PosteriorResult` and write it to disk via the CLI.

## Context (self-contained)
- `inference/decode.py`: `decode(fit: FitResult, data: PreparedDataset, engine=None) -> PosteriorResult`: builds `space` from `fit.spec`, runs `forward_backward`, `viterbi`, `segments`, criteria (E4-T05) → `PosteriorResult(times, filtered (T,N), smoothed (T,N), viterbi regime path (T,), segments, augmented, loglik, aic, bic, icl)`.
- `persistence/results.py`: `write_posterior(result, path)` → `posterior.parquet` with columns `time, filtered_0..N−1, smoothed_0..N−1, viterbi`; optional `augmented.parquet` (`time × S` smoothed) when `--augmented`; `segments.json`; `criteria.json`.
- CLI: `regimekit decode CONFIG --model DIR [--augmented]`: loads bundle (E8-T01 — until then, accept `--model` pointing at a `params.safetensors`+`spec.json` pair written by a temporary helper in this ticket; E8-T01 replaces the loader) and `prepared.parquet` from the output dir (or refetches when `--offline` is not set).

## Creates
- `src/regimekit/inference/decode.py`, writers in `persistence/results.py`, `decode` command
- `tests/integration/test_e6_t02_decode.py`

## Acceptance criteria
1. On a synthetic CSV project, `decode` writes `posterior.parquet` with `T` rows and `2N+2` columns; rows of `filtered_*`/`smoothed_*` sum to 1.
2. `segments.json` matches RLE of the `viterbi` column.
3. Exit code 4 if the engine returns non-finite loglik (crafted params).

## Out of scope
Bundle format (E8-T01); `Project` facade (E8-T02).

## Connections
- Depends on: E4-T05, E1-T02b
- Blocks: E8-T02
- Relates to: E8-T01 (loader swap)
