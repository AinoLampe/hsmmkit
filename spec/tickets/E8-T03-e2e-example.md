# E8-T03 — End-to-end example (`^GDAXI`, `^GSPC`) and tutorial notebook executed offline in CI

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | §1.4 criterion 2, NFR-03, NFR-07 | M6 | Appendix A, §5.8, §6.6 |

## Goal
Demonstrate the full flow `validate-config → fetch → fit → decode → forecast → backtest` on real indices and pin it as a reproducible, offline-replayable CI example.

## Context (self-contained)
- `examples/configs/dax_3state_gbm_nb.yaml` = Appendix A; `examples/configs/spx_3state_gbm_nb.yaml` = same with `^GSPC`.
- Fixtures for CI: a *short* offline snapshot (e.g. 3 years) produced once with `regimekit fetch` and stored under `tests/fixtures/snapshots/` **only if permitted by provider ToS**; otherwise generate a synthetic look-alike with the same schema (`simulate` from the Appendix A model) and document that CI uses synthetic data. Real-data runs are `-m live`.
- Tutorial notebook `examples/tutorial.ipynb` executed in CI with `--offline` fixtures (`nbclient`/`jupyter nbconvert --execute` via the `docs` nox session or a dedicated `examples` session); it walks through the Python API (E8-T02), plots regime probabilities and forecast fans (matplotlib is dev-only), and shows the backtest table.
- Reproducibility check: run `fit` twice on the snapshot with seed 42 and assert bitwise-equal `params.safetensors` (same hardware/backend).

## Creates
- `examples/configs/*.yaml`, `examples/tutorial.ipynb`, `tests/integration/test_e8_t03_e2e.py`, `tests/live/test_e8_t03_real_data.py` (`-m live`), nox `examples` session

## Acceptance criteria
1. Offline e2e test runs all six commands on the fixture and the results directory contains every artifact of the bundle (E8-T01) plus `posterior.parquet`, `forecast.parquet`, `backtest_metrics.parquet`, `strategy.parquet`, `report.md`, `quality.json`.
2. Bitwise reproducibility of `params.safetensors` across two fits (CPU).
3. Notebook executes in CI without errors in < 5 min.
4. Live test on `^GDAXI` and `^GSPC` passes when run manually with network.

## Out of scope
Final docs polish (E8-T04).

## Connections
- Depends on: E8-T02, E2-T02, E2-T04
- Blocks: E8-T04
- Relates to: §1.4 criterion 2
