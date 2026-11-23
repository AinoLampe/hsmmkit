# E10-T01 — Split guardrails: v0.2.0 golden values, neutral-name scanner, `core-only` CI job, ADR-0014

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | S | FR-G01, FR-G08, FR-G09, D23 | M7a | Part II §11.2 (D12, D22, D23), §15.8, §16.2, §18; Part I §6.4.6 |

## Goal
Install the measurements that make the decouple phase safe: frozen v0.2.0 golden values, a scanner that lists every finance term still in core modules, and a CI job that runs the core with the finance package blocked.

## Context (self-contained)
**Golden values.** On tag `v0.2.0` (the latest release; 0.1.0 plus E5-T06 and E9-T01), run the Appendix A pipeline on the offline fixtures (`--offline`, cassettes/CSV fixtures already used by E8-T03) and store under `tests/golden/v0_2_0/`: `prepared.npz` (`observations, dt, mask, cov_*`), `fit.npz` (`uparams`, `loglik`, `objective`), `posterior.npz` (`filtered, smoothed, viterbi`), `forecast.npz` (samples for seed 42, `h = 20`, `n_sim = 2000`), `backtest_metrics.parquet` (2 folds, reduced `n_sim`), plus `manifest.json` (jax/jaxlib/numpy versions, backend, git SHA). `tests/golden/test_e10_t01_v0_2_0.py` (marker `finance`) recomputes all of them and asserts **bitwise** equality (`np.array_equal`, `loglik ==`). The generator script `scripts/make_golden_v0_2_0.py` is committed; regenerating is only allowed on a toolchain change and must be recorded in `manifest.json`.

Note: `src/regimekit/estimation/recovery.py` (`NOMINAL_DT = 1/252`) and `utils/clock.py` (`TRADING_DAYS_PER_YEAR`) are known hits; they start on the allow-list tagged `E10-T03`.

**Neutral-name scanner** (`tests/unit/test_e10_t01_neutral_names.py`). Scope: every module under `src/regimekit/{config,data,model,statespace,engine,estimation,inference,evaluation,persistence,cli,utils}` and `project.py`, `__init__.py` (i.e. everything that will become `hsmmkit`). Uses `ast` + `tokenize`:
- identifiers (names, attributes, args, function/class names, dict keys in literals) matched case-insensitively against `price`, `prices`, `returns`, `log_return`, `symbol`, `ticker`, `trading`, `provider`, `strategy`, `sharpe`, `drawdown`, `annualiz`, `bars_per_day`;
- string constants, docstrings and comments matched against the same terms (Google `Returns:` section headers and the Python keyword `return` are excluded);
- numeric literal `252`.
Each hit is `path:line:term`. `tests/fixtures/neutral_names_allowlist.txt` lists accepted hits as `path:term  # E10-Txx` (line numbers not stored, so edits don't churn it). The test fails if a hit is not allow-listed and also if an allow-listed entry no longer occurs (stale entries must be removed). Every entry carries the ticket tag that will remove it (E10-T02b…E10-T07b); E10-T07b asserts the file is empty.

**Dependency test.** Extend the E1-T01 AST profile-rule test to the same scope (+ `cli/`, `project.py`, which were exempt in Part I): imports of `regimekit.finance` are only allowed in a temporary list `tests/fixtures/finance_import_allowlist.txt` with the same ticket-tag rule (emptied by E10-T07b).

**`core-only` job.** `tests/conftest.py` gains an opt-in (`HSMM_CORE_ONLY=1`) meta-path finder that raises `ImportError("finance blocked")` for `regimekit.finance*` and `yfinance`, `fredapi`, `pandas_datareader`, `curl_cffi`. All finance tests get marker `finance` (moved/marked in this ticket). nox session `core` runs `pytest -m "not finance and not live and not slow and not benchmark"` with the flag. CI job `core-only` on PRs: until E10-T07b it is allowed to fail but publishes the failing test IDs as an artifact; E10-T07b makes it required.

**ADR-0014** (`docs/adr/0014-two-repo-split.md`, status *proposed*, accepted in E10-T08): context (Part II v1.2), decision (D12/D22/D23 verbatim; repository `RegimeDetectionAndForecast` keeps package `regimekit`, new repository `hsmmkit`), consequences (two release trains, `downstream` job, history-preserving extraction, ADR-0012 superseded at acceptance).

## Creates
- `scripts/make_golden_v0_2_0.py`, `tests/golden/v0_2_0/*`, `tests/golden/test_e10_t01_v0_2_0.py`
- `tests/unit/test_e10_t01_neutral_names.py`, `tests/fixtures/neutral_names_allowlist.txt`, `tests/fixtures/finance_import_allowlist.txt`
- `tests/conftest.py` (core-only finder), `noxfile.py` (`core` session), `.github/workflows/pr.yml` (`core-only` job)
- `docs/adr/0014-two-repo-split.md`

## Acceptance criteria
1. Golden test passes bitwise on the current `main`; altering one bit of a stored array makes it fail.
2. Adding `def price(): ...` to `src/regimekit/engine/base.py` fails the scanner with `engine/base.py:<line>:price`; a docstring section `Returns:` does not.
3. Every allow-list entry carries a ticket tag from `E10-T02b … E10-T07b`; a stale entry fails the test.
4. `HSMM_CORE_ONLY=1 python -c "import regimekit.finance"` raises `ImportError`; `nox -s core` runs and the job uploads the list of failing test IDs.
5. ADR-0014 exists with status *proposed*.

## Tests
Scanner unit tests on synthetic source snippets (identifier, string, comment, docstring section, literal `252`); golden bitwise test; core-only finder test.

## Out of scope
Removing any finance term (E10-T02b…E10-T07b); the extraction itself (E10-T08).

## Connections
- Depends on: Part I complete (E8-T04, E9-T01…T04)
- Blocks: E10-T02a
- Relates to: E1-T01 (AST test), E10-T07b (allow-lists emptied), E10-T09 (golden values reused)
