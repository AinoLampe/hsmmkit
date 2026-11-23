# E16-T04 — Entry-point plug-in discovery, `hsmmkit new-family` scaffold, `plugin` CI job, ADR-0017

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-PL01, FR-PL02, FR-PL03, NFR-14, D15 | M12 | Part II §13.1 FR-PL, §15.1 (`plugins.py`, `testing/`), §16.1 (`copier`), §16.5 |

## Goal
Let third-party packages (including `regimekit`) contribute components by installation alone, and make writing a compliant plug-in a five-minute task.

## Context (self-contained)
- **Discovery** (`hsmmkit.plugins.Registry._discover`, E10-T02a hook): on the first `get/names/entries` call, iterate `importlib.metadata.entry_points(group=g)` for `g ∈ {hsmmkit.emission, hsmmkit.duration, hsmmkit.metric, hsmmkit.source, hsmmkit.clock, hsmmkit.transform, hsmmkit.init, hsmmkit.preset, hsmmkit.splitter, hsmmkit.estimator, hsmmkit.ppc_statistic}`; each entry point name = component name, value = `module:object`; load lazily (store a loader, import on first `get` of that name — `names()` does not import); provenance = distribution name + version. Duplicate names across distributions → `ConfigError` naming both distributions (no silent override); built-ins win only if the plug-in sets `override = true` in an `[tool.hsmmkit]` table — **not supported** in 0.4.0 (documented), so duplicates are always errors. A failing plug-in import raises at `get` time with the distribution name; `names()` still lists it with a `(failed to load)` marker in `hsmmkit plugins list`.
- CLI `hsmmkit plugins list [--kind K]` shows name, kind, distribution, version, load status.
- **Contract suites as public API** (FR-PL02): `hsmmkit.testing` exports `family_contract`, `duration_contract` (new: pmf sums to 1, mean vs `mean_sojourn`, hazard consistency, censoring survival, phase/explicit agreement for families supporting both), `metric_contract` (new: name/needs declared, deterministic, handles NaN inputs per docs), `source_contract`, `clock_contract`; documented as semver-stable from 0.3.0 (NFR-14, ADR-0025 in E17-T03).
- **Scaffold** (`hsmmkit new-family NAME [--kind emission|duration] [--dir PATH]`): runs a `copier` template from `templates/family-plugin/` (shipped as package data; `copier` imported lazily — dev extra `scaffold`, documented): `pyproject.toml` with the entry point, `src/<pkg>/family.py` skeleton implementing the protocol with TODOs, `tests/test_contract.py` registering a `FamilyCase`, README. The generated package's tests pass out of the box with a placeholder Gaussian-like implementation (so users start green).
- **Template plug-in** for CI: `examples/plugins/hsmmkit-vonmises/` — von Mises emission (circular, `support="real"` with a domain note), `scale_by_dt=False`, sampler via rejection or `jax.random` if available, `cdf` omitted.
- **CI job `plugin`**: build the example plug-in wheel, install into a fresh venv with `hsmmkit`, run `pytest --pyargs hsmmkit_vonmises` (contract suite) and a config using `emission.family: vonmises` end-to-end (`fit`, `decode`).
- **ADR-0017** (`docs/adr/0017-entry-points.md`, accepted).

## Creates
- `src/hsmmkit/plugins.py` (discovery), `src/hsmmkit/testing/{duration_contract,metric_contract}.py`, CLI `plugins`, `new-family`
- `templates/family-plugin/`, `examples/plugins/hsmmkit-vonmises/`
- `tests/integration/test_e16_t04_discovery.py`, `.github/workflows/pr.yml` (`plugin` job)
- `docs/guide/extending/plugins.md`, `docs/adr/0017-entry-points.md`

## Acceptance criteria
1. The von Mises plug-in, installed into a temp venv, is discovered, passes `family_contract`, and fits/decodes via config.
2. Two distributions registering the same name → `ConfigError` naming both; a plug-in with an import error is listed as failed and raises only when used.
3. `hsmmkit new-family foo` produces a package whose tests pass immediately.
4. `names()` does not import plug-in modules (asserted with `sys.modules`).
5. All built-in components pass their contract suites (duration and metric suites newly parametrized over built-ins).

## Tests
Integration (temp venv), discovery unit tests, scaffold test.

## Out of scope
Switching `regimekit` to entry points (E16-T06).

## Connections
- Depends on: E11-T01
- Blocks: E16-T06, E17-T03
- Relates to: E10-T02a, E10-T03, E10-T04a
