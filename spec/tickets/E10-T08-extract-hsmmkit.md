# E10-T08 — Extract the `hsmmkit` repository with history, rename, CI (incl. `downstream`), release `hsmmkit` 0.1.0

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-G09, FR-G08, NFR-18, D22, D23 | M7b | Part II §15.1, §15.8, §16.5, §19 #22–23; overview §2.3 rename map |

## Goal
Create the new library repository from the decoupled core with full git history, rename it to `hsmmkit`, wire CI including the nightly downstream check against `RegimeDetectionAndForecast`, and publish `hsmmkit` 0.1.0.

## Context (self-contained)
**Extraction** (documented, reproducible script `scripts/extract_hsmmkit.sh`, run once on a fresh clone of `RegimeDetectionAndForecast` at the E10-T07b merge commit, kept in the new repo under `tools/` for the record):
1. Pass 1 (keep): `git filter-repo --path src/regimekit --path tests --path docs --path benchmarks --path scripts --path noxfile.py --path pyproject.toml --path .github --path mkdocs.yml --path CONTRIBUTING.md --path LICENSE --path README.md --path .pre-commit-config.yaml --path .python-version --path CLAUDE.md --path .claude --path spec --path OVERVIEW.md --path CHANGELOG.md --path changelog.d --path towncrier.toml --path uv.lock`. Pass 2 (drop finance; `filter-repo` cannot mix include and exclude in one run): `git filter-repo --invert-paths --path src/regimekit/finance --path tests/finance --path tests/fixtures/cassettes --path tests/live --path tests/golden/v0_2_0 --path-glob 'docs/guide/finance*'`.
2. `--path-rename src/regimekit/:src/hsmmkit/`.
3. Commit "rename regimekit → hsmmkit" applying the rename map: package and imports, distribution name, console script `hsmmkit` (from `regimekit-core`), `RegimekitError → HsmmkitError`, `REGIMEKIT_TYPECHECK → HSMMKIT_TYPECHECK`, `HSMM_CORE_ONLY` flag and finder removed (nothing to block), default bundle suffix `.hsmm`, logger names, docs site name, test names stay (`test_e4_t03a_*` etc.).
**pyproject**: `name = "hsmmkit"`, version `0.1.0`, dependencies = Part I runtime set minus provider packages; extras `gpu`, `mcmc` (unchanged until E10-T11); `[project.scripts] hsmmkit = "hsmmkit.cli.app:main"`; no `regimekit` anywhere.
**CI**: workflows carried over (`pr.yml`, `nightly.yml`, `gpu.yml`), plus `downstream` in `nightly.yml` (and `workflow_dispatch`): checkout `AinoLampe/RegimeDetectionAndForecast@main` (needs a read token secret if that repository is private), install `hsmmkit` from this repo's HEAD in editable mode into its env overriding the pin, run `pytest -m "not live"` of `RegimeDetectionAndForecast` (golden + finance + contract tests). Failure opens/updates a GitHub issue labelled `downstream` (via `gh`).
**Docs**: site renamed; ADR-0001…0013 kept (history); ADR-0014 status → accepted, ADR-0012 marked "superseded by 0014", ADR-0013 marked "amended by 0015"; `docs/guide/index.md` states domain neutrality and points to `regimekit` as an example extension.
**Golden values**: `hsmmkit` keeps the core golden/reference tests (re-expressed with `gaussian` in E10-T06); they must pass unchanged — no regeneration.
**Cloud-session constraints** (Claude Code cloud sessions push through a GitHub proxy that rejects tag pushes and branch deletions): the extraction pushes the rewritten history as branch `main` to the empty `hsmmkit` repository; tags carried over from `RegimeDetectionAndForecast` (`v0.1.0`, `v0.2.0`) are **not** pushed and are not needed in `hsmmkit`. `CLAUDE.md` is rewritten for the new repo (package name, commands, Part II overview import `@spec/tickets/10-OVERVIEW-part2-build-order-and-conventions.md`, `extensions.md` v1.2 as ticket source) and `spec/STATUS.md` restarted with the `hsmmkit` tickets only; the Part I tickets stay in `spec/tickets/` as history.
**Release**: tag `v0.1.0` (created by the maintainer on GitHub, not from a cloud session), GitHub release with sdist + wheel (`uv build`), TestPyPI upload (PyPI trusted publishing is E17-T02). `regimekit` consumes it via a git tag or TestPyPI in E10-T09.

## Creates
- new repository `hsmmkit` (history-preserving), `tools/extract_hsmmkit.sh`, `.github/workflows/nightly.yml` (`downstream` job), `docs/adr/0014-two-repo-split.md` (accepted), `CHANGELOG` entry 0.1.0
- `tests/unit/test_e10_t08_package.py` (import, metadata, no `regimekit` strings)

## Acceptance criteria
1. `git log --follow src/hsmmkit/engine/expanded_hmm.py` shows the Part I commits (E4-T03a/b) with original authors and dates.
2. Full test-suite green; `nox -s lint types tests docs` green; benchmark baseline carried over.
3. `python -c "import hsmmkit, sys; assert not any(m.startswith(('regimekit','yfinance','fredapi','pandas_datareader')) for m in sys.modules)"` passes; `grep -ri regimekit src/ docs/guide` finds only the "example extension" mention.
4. `downstream` job runs against `RegimeDetectionAndForecast` (expected red until E10-T09 merges; green afterwards — re-run recorded in E10-T09).
5. `v0.1.0` tag and release artifacts exist; `pip install` of the wheel in a fresh env and `hsmmkit validate-config` on a core fixture work.

## Tests
Package/metadata test; full suite; release smoke install.

## Out of scope
Removing the core from `RegimeDetectionAndForecast` (E10-T09); new extras (E10-T11); PyPI/conda-forge (E17-T02).

## Connections
- Depends on: E10-T07b
- Blocks: E10-T09, E10-T10, E10-T11, E11-T01, E14-T03, E15-T01
- Relates to: E10-T01 (ADR-0014), E17-T02
