# E15-T04 — `hsmmkit stream` CLI (stdin/CSV tail → JSONL events) and `Project.online()`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | FR-C02, FR-C03, FR-ON01…ON04 | M12 | Part II §14.11; Part I §5.9 (CLI conventions), E8-T02 |

## Goal
Expose online inference on the command line and in the Python facade so a running process can consume observations and emit regime probabilities and alerts.

## Context (self-contained)
- `hsmmkit stream CONFIG --model PATH [--input -|FILE] [--follow] [--state PATH] [--emit filtered,smoothed,events] [--every k]`:
  - input: CSV lines with header (`time` optional, observation columns, covariate columns) from stdin (`-`) or a file; `--follow` tails a growing file (poll interval 0.5 s, stops on SIGINT/SIGTERM gracefully after flushing).
  - each processed row → JSONL on stdout: `{"t", "time", "filtered": [...], "smoothed_lag": [...] | null, "events": [...]}`; `--every k` emits every k-th row (events always emitted).
  - `--state PATH`: load state at start if it exists (resume), save on exit and every `online.checkpoint_every` rows (default 1000).
  - covariates are raw values; `OnlineCovariatePreparer` (E15-T01) applies lags/transforms/standardization; missing covariate column → exit 3 with the column name.
  - exit codes per overview §5.4; malformed row → logged and skipped unless `--strict` (then exit 3).
- `Project.online(fit, *, initial: PreparedDataset | None = None, lag: int | None = None) -> OnlineSession` with `push(row: dict | pd.Series) -> dict`, `push_many(df) -> pd.DataFrame`, `events`, `save(path)`, `forecast(horizon)`.
- `regimekit` inherits the command via the app factory (E10-T07b); no code there (covered by `downstream`).

## Creates
- `src/hsmmkit/cli/stream.py`, `project.py` (`online`), `online/session.py`
- `tests/integration/test_e15_t04_stream.py` (piped fixture via `CliRunner` and subprocess)

## Acceptance criteria
1. Piping the fixture CSV through `hsmmkit stream` yields JSONL whose filtered probabilities equal `Project.online().push_many()` and the offline filter bitwise.
2. Interrupting after N rows with `--state` and restarting yields the same output for the remaining rows as an uninterrupted run.
3. Exit codes: missing model → 2, missing covariate column → 3; `--strict` on a malformed row → 3.
4. `Project.online` is importable and usable without the CLI.

## Tests
Integration with a piped fixture; resume test; exit codes.

## Out of scope
Network ingestion (Kafka, websockets) — out of scope for the library.

## Connections
- Depends on: E15-T03
- Blocks: E17-T03
- Relates to: E8-T02, E10-T07b
