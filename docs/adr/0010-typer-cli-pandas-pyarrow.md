# ADR-0010: typer CLI; pandas in the data layer; pyarrow parquet

## Context

regimekit needs a command-line interface (`fetch`, `decode`, `fit`, `forecast`, `backtest`, `compare`,
`selftest`, `validate-config`), a tabular data layer for provider ingestion and preparation, and a local
caching format for fetched price/covariate series.

## Decision

The CLI is built with `typer` (`cli/app.py`). The data layer (`data/`) uses `pandas` for tabular
manipulation (alignment, resampling, lagging) before conversion to the JAX-facing `PreparedDataset` contracts.
The on-disk cache format is Parquet via `pyarrow` (`data/cache.py`).

## Consequences

- `typer`'s type-hint-driven CLI definitions keep command signatures close to their Python implementations and
  give free `--help` generation.
- `pandas` is confined to `data/` (and `finance/`, which sits on top of it); the layer rule (overview §4.3)
  keeps it out of `model/`, `statespace/`, `engine/`, `estimation/`, `inference/`.
- Parquet gives compact, typed, columnar local caching with good pandas/pyarrow round-tripping, and is the
  format E2-T02/E8-T03 fixtures use.

## Status

accepted
