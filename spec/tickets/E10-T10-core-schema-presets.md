# E10-T10 — Complete `hsmmkit` schema (all Part II keys) with feature gates; JSON-Schema export with registry enums; presets registry

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-G05, FR-G06, FR-IO05 | M8 | Part II §15.4 (hsmmkit schema 1), §15.6, Appendix H; overview §5.11 |

## Goal
Make the configuration schema final for 0.2.0–0.5.0 up front, so later tickets only remove feature gates, and ship the preset registry with the first core presets.

## Context (self-contained)
**Schema additions** (`hsmmkit.config.schema`, still `schema_version: 1`; all new keys optional with defaults that keep current behaviour):
| Key | Type / default |
|---|---|
| `data.source.sequence_column` | `str | None = None` |
| `data.sequences` | `{group_by: str | None = None, bucket_edges: list[int] = [256, 512, 1024, 2048, 4096]}` (strictly increasing, ≥ 1) |
| `data.labels` | `None | {source: "column" | "file", column: str | None, path: Path | None, allowed_sets: bool = False, boundaries: {must_switch_at: list[int], must_not_switch_at: list[int]} | None}` |
| `model.topology` | `"full"` (default) `| "left_to_right" | "cyclic" | {banded: int ≥ 1} | {matrix: list[list[int]]}` |
| `model.left_censored` | `bool = False` |
| `model.initial_groups` | `"shared" | "per_group" = "shared"` (FR-SQ02) |
| `model.emission` | `covariance: "diag" | "full" = "diag"`, `mixture: {components: int ≥ 1, base: str} | None` |
| `model.duration` | either `family` (Part I) or `per_state: list[{family, params}]` (length `n_states`), `duration_dependent_jumps: bool = False` |
| `estimation.method` | `"map" | "em" | "gem" | "vb" | "mcmc" = "map"`; sub-sections `em: {tempering: {enabled, beta0, steps}}`, `gem: {tempering: …}`, `vb: {prune_below: float = 0.01, max_iter}`, `mcmc: {sampler: "nuts" | "gibbs", chains: 4, warmup: 1000, draws: 1000}` |
| `estimation.init` | registered names; core registers `labels` now (gated) |
| `online` | `None | {lag: int ≥ 0, alerts: list[{regime, p_on, p_off} | {regime, remaining_duration_below}]}` |
| `evaluation` | `cv: {splitter: "sequence_kfold" | "walk_forward", folds, seed} | None`, `grid: {n_states: list[int], r: list[list[int]], emission: list[str], estimator: list[str]} | None`, `boundary_tolerance: int = 5` |
Validation rules (Part II §15.4): family `support` vs data support (at build); topology rows non-empty unless intended absorbing (warning listing the states); `duration_dependent_jumps` requires `engine: explicit`; `per_state` requires `engine: explicit`; `method: vb|mcmc` warns on non-conjugate components (warning text lists them; conjugacy known from family `conjugate_prior()`); `labels` require `n_states`; `p_off < p_on`; `banded k < n_states`; matrix shape `(N, N)` with zero diagonal.
**Feature gates** (`hsmmkit.config.gates`): `GATES: dict[str, tuple[Callable[[ProjectConfig], bool], str]]` mapping a key path to (predicate "feature used", ticket ID); `check(cfg)` raises `ConfigError(path, "not available before <ticket>")` at model build (not at validation, so presets and JSON Schema validate). Initial rows = overview §5.11 table.
**JSON Schema**: `hsmmkit config-schema OUT.json [--config-class dotted.path]` — registry-validated fields get `enum` from `REGISTRY.names(kind)` at export (includes extension components when the extension is installed).
**Presets** (registry kind `preset`): a preset = `{name, description, yaml: str, fixture: Callable[[int seed], dict] | None}`. CLI `hsmmkit presets list | show NAME | write NAME DIR` (`write` emits the YAML and, if a fixture generator exists, the synthetic data file referenced by the YAML). Core presets now: `core/segmentation_gaussian_ltr` (Appendix H verbatim; fixture generator: 50 sequences of 3-dim Gaussian windows with left-to-right 4-state structure, lengths 150–400, partial labels), `core/two_regime_gaussian` (Appendix E in Gaussian form under `calendar` with the daily grid; fixture: simulated series). `core/counts_poisson_hazard`, `core/msar2_hamilton` come in E16-T05.

## Creates
- `src/hsmmkit/config/{schema,gates}.py`, `src/hsmmkit/presets/{__init__,core/segmentation_gaussian_ltr.yaml,core/two_regime_gaussian.yaml,fixtures.py}`, CLI `presets`, `config-schema --config-class`
- `tests/unit/test_e10_t10_schema.py`, `tests/unit/test_e10_t10_gates.py`, `tests/integration/test_e10_t10_presets.py`
- `docs/guide/configuration.md` (generated reference updated)

## Acceptance criteria
1. Appendix H validates; building its model raises the gate error for the first gated key in deterministic order (`data.sequences` → E12-T01a).
2. `hsmmkit presets show core/segmentation_gaussian_ltr` prints Appendix H; `presets write` produces YAML + parquet fixture that `validate-config` accepts.
3. JSON Schema validates Appendix H; with `regimekit` installed in the test env (dev extra), `--config-class regimekit.config.FinanceConfig` validates `regimekit`'s Appendix A v2 and lists `gbm` in the family enum.
4. Every validation rule above has a failing fixture asserting the `ConfigError.path`.
5. Existing Part I-derived core configs build unchanged (defaults preserve behaviour).

## Tests
Schema rules, gate mechanism (a test per gate row that the gate fires), preset CLI, JSON Schema.

## Out of scope
Implementing any gated feature; remaining presets and tutorials (E16-T05).

## Connections
- Depends on: E10-T08
- Blocks: E12-T01a, E12-T03, E12-T04, E16-T05
- Relates to: E10-T02b, every ticket that removes a gate
