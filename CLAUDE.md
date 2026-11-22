# CLAUDE.md — RegimeDetectionAndForecast (Python package `regimekit`)

This repository is called **RegimeDetectionAndForecast**. The Python package, distribution and CLI inside it are
called **`regimekit`**: a JAX-based Hidden Semi-Markov Model framework for market-regime detection and
probabilistic return forecasting (Python 3.13, uv, JAX + Equinox + Optax, pydantic v2, typer). It is built ticket
by ticket from a written specification.

- **Part I** (E0–E9, 63 tickets) is done and released: `v0.1.0` (2026-10-24) and `v0.2.0` (2026-11-01, adds E5-T06
  and E9-T01).
- **Part II** (E10–E17, 66 tickets) is the current work. It splits the domain-neutral core into a new library
  **`hsmmkit`** (its own repository). This repository stays the finance application: package `regimekit` that
  depends on `hsmmkit`.

Both ticket overviews are binding and imported here. Part I conventions stay in force unless the Part II overview
changes them.

@spec/tickets/00-OVERVIEW-build-order-and-conventions.md
@spec/tickets/10-OVERVIEW-part2-build-order-and-conventions.md

This file adds (a) the Part II scope rules for this repository (§0), (b) the precedence rules (§1),
(c) **harmonization decisions that resolve naming and placement conflicts between concept and tickets** (§3),
and (d) the working rules for autonomous implementation.

## 0. Part II scope in this repository

Every Part II ticket names its repository in the `Repo` column (Part II overview §2.1).

| Repo value | Tickets | Where |
|---|---|---|
| `RegimeDetectionAndForecast*` | E10-T01, E10-T02a, E10-T02b, E10-T03, E10-T04a, E10-T04b, E10-T05, E10-T06, E10-T07a, E10-T07b | here, **before** the split (phase M7a, "decouple") |
| `RegimeDetectionAndForecast` | E10-T09, E11-T07, E11-T09, E16-T06, E17-T05 | here, **after** the split |
| `hsmmkit` | all other E10–E17 tickets | **not here**. If asked to implement one, stop and say it belongs to the `hsmmkit` repository. |

- **E10-T08 (extraction)** starts from a clone of this repository but pushes to the `hsmmkit` repository. Run it only
  when the owner asks for it in the current session and the `hsmmkit` repository is attached. It is a one-way step:
  confirm the commit to extract (the E10-T07b merge commit) with the owner before running `git filter-repo`.
- **Picking the next ticket.** Follow the Part II build order (Part II overview §6) and the dependency graph (§7).
  The next ticket is the first `todo` row in `spec/STATUS.md` whose dependencies are `done`. A dependency on an
  `hsmmkit` ticket counts as done only when STATUS records the `hsmmkit` release that contains it.
- **Phase M7a rule (E10-T01…T07b).** These tickets only move, rename and add extension points. Nothing gains new
  behaviour. The v0.2.0 golden values stay bitwise identical at every merge (Part II overview §5.7, §10).
- **After the split.** E10-T09 deletes the core packages from this repository and makes `regimekit` depend on
  `hsmmkit`. In the same ticket, rewrite this file for the post-split state: drop the rules that only concern
  core code, which then live in the `hsmmkit` repository's `CLAUDE.md` written by E10-T08.

## 1. Sources of truth and precedence

| Path | Role |
|---|---|
| `spec/tickets/<ID>-*.md` | The unit of work, Part I (`E0`–`E9`) and Part II (`E10`–`E17`). Each ticket is self-contained (Context, Creates, Acceptance criteria, Tests). |
| `spec/tickets/00-OVERVIEW-…md` | Part I conventions (§4), build order, index, Part II amendments (§8), recovery amendment (§9). |
| `spec/tickets/10-OVERVIEW-part2-…md` | Part II: repositories and phases (§2), conventions and tolerances (§5), feature gates (§5.11), build order (§6), dependency graph (§7), index (§8), relation to Part I tickets (§9), guardrails (§10). |
| `spec/concept/regimekit_concept.md` | Part I concept **text v1.0**. "Concept v1.1" = this text + Part I overview §8. Read only when a ticket is ambiguous. |
| `spec/concept/extensions.md` | Part II concept **v1.2** ("Approved for ticket derivation"). It is the concept for every E10–E17 ticket. Its "Changes in 1.2" box lists the decisions and corrections relative to draft 1.1. |
| `spec/STATUS.md` | Progress ledger you maintain (§12). |

Precedence for **Part II tickets**, highest first:
1. **§3 of this file** (harmonization decisions).
2. **Part II overview** (§5 conventions and tolerances, §10 guardrails).
3. **`extensions.md` v1.2** for Part II semantics; **Part I concept** for every Part I semantic that Part II does not
   change (formulas, defaults, tolerances of existing components).
4. **Ticket** for everything the concepts leave open: exact signatures, file names, column names, test design.

Precedence for **Part I tickets** (only E5-T05b is still open; see STATUS): unchanged. §3, then Part I overview
§8/§9, then Part I concept, then ticket.

If a ticket *contradicts* the concept on semantics (not a naming or detail refinement) and §3 doesn't cover it,
the concept wins and the ticket is fixed. Implement the concept's version, correct the ticket file in a separate
commit `docs(spec): align <ID> with concept §x.y`, and log it in STATUS → "Spec fixes". If it is unclear which
reading is right, stop (§11).

Never edit `spec/concept/*`. Never edit the overviews. Ticket files only as described above.

Known slips in the Part I concept text: §0 and §7 say "Section 6.5" / "6.5.4" for testing and recovery; correct
is **§6.4 / §6.4.4**. The Part I concept still shows v1.0 names (e.g. `PreparedDataset.returns`); §3.1 lists the
v1.1 names. `extensions.md` v1.2 already contains its own corrections (Hamilton MS-AR uses a time-step memory
expansion; fixed-lag smoothing is compared with the offline smoother on the same prefix).

Naming in the tickets: "repository"/"repo" + `RegimeDetectionAndForecast` means this repository. Everything else
called `regimekit` means the Python package, distribution or CLI.

## 2. Commands

```bash
uv sync --all-groups                                     # never pip
uv run nox -s lint types tests                           # full gate; must be green before merging a ticket
uv run pytest tests/<layer>/test_e10_t02a_*.py -x -q     # fast loop on the current ticket
uv run ruff format . && uv run ruff check --fix .
uv run mypy
uv run python scripts/check_package_coverage.py          # per-package ≥ 85 % (after a --cov run)
uv run nox -s docs                                       # mkdocs build --strict
uv run nox -s recovery_smoke                             # nightly smoke: 1 scenario at T=5000
uv run nox -s recovery                                   # -m slow, full suite (weekly, hours)
uv run nox -s bench                                      # -m benchmark
# Part II, from E10-T01 on:
uv run nox -s core                                       # core suite with regimekit.finance blocked (HSMM_CORE_ONLY=1)
uv run pytest tests/golden/test_e10_t01_v0_2_0.py -q     # v0.2.0 golden values, bitwise
uv run pytest tests/unit/test_e10_t01_neutral_names.py -q
```

- The nox `tests` session sets `REGIMEKIT_TYPECHECK=1` and `JAX_PLATFORMS=cpu`. The default pytest run excludes
  `slow`, `live`, `benchmark` and `gpu` (E0-T02 `addopts`). From E10-T01 on, finance tests also carry the marker
  `finance`.
- Add dependencies only with `uv add` / `uv add --group dev`. Never edit `uv.lock` by hand.

## 3. Harmonization decisions (override ticket and concept text)

### 3.1 v1.1 names — never reintroduce the v1.0 names

| v1.0 / conflicting text | Canonical |
|---|---|
| `PreparedDataset.returns` (concept §5.4, E9-T04 text) | `PreparedDataset.observations`. Config keys `data.preparation.returns` / `missing_returns` stay in `schema_version: 1` (Appendix A); E10-T02b migrates them to v2 names. Local variables for sampled log-returns may be called `returns`. |
| "`dt`: Δ_t in years" (E1-T01 comment) | `dt` is in **clock units**: years under `trading`/`calendar`, steps (≡ 1) under `steps`. Same for all parameters: annualized or per step, never both. |
| E3-T04 "Parameters (annualized)", init formulas with `/252` | Explicit duration families are in clock units too; init goes through `days_to_steps` (§3.2) and `init_from_mean_sojourn` (§3.3). |
| `mean_sojourn_years(theta)` | `mean_sojourn(theta, nominal_dt)` |
| `estimation/fit.py` | `estimation/map.py`; `from regimekit.estimation import fit` |
| `data/providers/` | `finance/providers/` |
| `model/params.py` | package `model/params/` (`bijectors`, `priors`, `spec`, `manager`; `__init__` re-exports) |
| `@jaxtyped(typechecker=beartype)` | `from regimekit.utils.typing import typechecked` (active only with `REGIMEKIT_TYPECHECK=1`) |
| `Estimator.fit(spec, data, cfg, key)` | `fit(spec, data, cfg, seed: int) -> FitResult` |
| `jump_probs(theta, z_pi, memory_index: StateIndex)` | `jump_probs(theta, z_pi, memory: MemoryLayout)`; `MemoryLayout` lives in **`model/transition.py`**. |
| `StateIndex.memory_tuple_index()` (E1-T05) vs `row_of_state` (E4-T02a/b) | Array field **`row_of_state (S,)`**; no method. All other index arrays as placed by E4-T02a/b. |
| `InferenceEngine.loglik -> float` | Engine methods: `loglik`, `forward`, `forward_backward`, `viterbi`, `approx`. `loglik` and `Posterior.loglik` are 0-d `Array`s (pytree leaves). `PosteriorResult.loglik` and `FitResult.loglik` are Python `float`. |
| `Posterior.elapsed_posterior()` "dispatches on engine tag" | `Posterior` has a static field `engine: Literal["expanded", "explicit"] = "expanded"`. |
| `PosteriorResult.viterbi` | The **regime** path `(T,)`; expanded paths stay inside `augmented`. |
| `data.irregular` (E4-T06a) | No such field. Use `utils.clock.is_irregular(dt)` (`max/min > 1.5`). |

### 3.2 Placement decisions (keep the layer, profile and dependency rules intact)

| Conflict | Decision |
|---|---|
| E6-T04a puts `future_grid` in `data/calendar.py`, but `inference/` may not import `data.calendar`. | **`regimekit/utils/clock.py`** holds `TRADING_DAYS_PER_YEAR = 252`, `CALENDAR_DAYS_PER_YEAR = 365.25`, `future_grid(...)`, `days_to_steps(...)` and `is_irregular(dt)`. `data/calendar.py` imports from it. **From E10-T03 on:** `TRADING_DAYS_PER_YEAR` and the trading branches of `future_grid`/`days_to_steps` live only in `regimekit/finance/clocks.py` (registered `trading` clock); `utils/clock.py` keeps the clock-neutral parts and dispatches through the clock registry. |
| Literal `252` / `365.25`. | Until E10-T03: no literal outside `utils/clock.py`. From E10-T03: `252` only in `finance/clocks.py`; `365.25` only in `utils/clock.py`. Branching on the clock *name* is allowed only in `utils/clock.py`, `data/calendar.py` and the clock registry. |
| E2-T05 `CachedProvider(inner: DataProvider)` would make `data/` import `regimekit.finance`. | `data/cache.py` declares its own structural Protocol `CacheableProvider`. **E10-T04b** moves the cache to `finance/cache.py`. |
| E0-T01 `.gitignore` has `*.parquet`; fixtures commit parquet files. | `.gitignore` adds `!tests/fixtures/**/*.parquet`. |
| Recovery harness location. | **`src/regimekit/estimation/recovery.py`** (pure, no pytest, no file I/O). Its `NOMINAL_DT = 1/252` is derived from the `trading` clock from E10-T03 on. |

### 3.3 Unowned deliverables and protocol completions

- **Config → model assembly.** `src/regimekit/model/build.py`: `build_model_spec(cfg, data) -> ModelSpec`, with
  name registries `EMISSION_FAMILIES`, `DURATION_FAMILIES`. **From E10-T02a on** these dicts are thin views over
  `regimekit.plugins.REGISTRY`; new lookups go through the registry only.
- **`DurationFamily`** additionally has `supports_explicit: bool`, `init_from_mean_sojourn(mean_steps, nominal_dt)`
  and the per-step wrapper `advance_prob_t`. **`TransitionModel`** gets `jump_probs_t`. Phase families size their
  rate arrays by `n_rates` (`N`, or `N(N−1)` for `pair` conditioning).
- **Class names:** `GBMFamily`, `StudentTFamily`; `GeometricDuration`, `NegBinDuration`, `PoissonDuration`,
  `GammaDuration`, `WeibullDuration`, `LogNormalDuration`, `NonparametricDuration`; `JumpTransition`;
  `ExpandedHMMEngine`, `ExplicitDurationEngine`; `MonteCarloForecaster`. Part II adds `GaussianFamily` (E10-T06;
  `GBMFamily` becomes its finance subclass). Registry keys = config names.
- **ParamSpec names** are `<component>.<config key>`: `emission.{mu,sigma,gamma,delta}` (GBM; kept unchanged by
  the E10-T06 alias), `emission.{mean,scale,gamma,delta}` (Gaussian), `emission.{m,s,nu,gamma,delta}` (Student-t),
  `duration.{rate,c}` (phase), `duration.{kappa,shape,scale,mu_log,sd_log,logits,c}` (explicit),
  `transition.{a,b}` or `transition.pi_rows` (no covariates), `initial.logits`.

### 3.4 Config fields

Part I fields introduced after E1-T02a (keep): `estimation.chunk_size`, `estimation.parallel_scan`, the
covariate-forecaster sub-keys, `evaluation.backtest.n_sim`, `evaluation.strategy.cap`/`allow_short`,
`model.emission.per_series`.

Part II: E10-T02b splits the schema into the neutral `ProjectConfig` (no `profile`) and `FinanceConfig`
(`schema_version: 2`, automatic v1 → v2 migration). A Part II ticket adds a config field to `FinanceConfig` when it
is finance-specific, to `ProjectConfig` otherwise. Each addition updates the JSON-Schema test and the
resolved-dump test in the same ticket. Appendix A (v1) must keep validating through the migration.

### 3.5 RNG

- `key_for(seed, *ids)` takes **ints only**. Purpose ids are named int constants in `utils/rng.py`.
- `MonteCarloForecaster.forecast(..., seed: int)`. Per-origin seeds use `utils.rng.derive_seed(seed, *ids) -> int`.

### 3.6 ADR numbers

| ADR | Content | Written by |
|---|---|---|
| 0001–0010 | Concept §6.7 baseline | E0-T04 |
| 0011 | safetensors bundle format | E8-T01 |
| 0012 / 0013 | core/profile split / `steps` clock (superseded by 0014 / amended by 0015) | E0-T04 |
| 0014 | Two-repository split (this repository + `hsmmkit`) | E10-T01 (accepted in E10-T08) |
| 0015 | Clock registry; `trading` in `regimekit` | E10-T03 |
| 0016 | Registries and config extension instead of `profile` | E10-T02b |
| 0017–0021 | Part II, written in the `hsmmkit` repository (plan: Part II overview §5.10) | `hsmmkit` tickets |
| **0022** | in-scan emissions, annualization exception | E9-T03 (exists) |
| **0023** | endogenous covariates | E9-T04 (exists) |
| 0024–0027 | Part II, written in the `hsmmkit` repository | `hsmmkit` tickets |
| 0028+ | ADRs raised during implementation (next free number; check both repositories) | any |

### 3.7 Index and range conventions

- Code index `t` (0-based) = math step `t+1`. `A_t` moves from code index `t−1` to `t`; covariate row `t` is the
  lagged value usable at step `t` (known at the end of `t−1`).
- Segments `(start, end, regime)`: `end` **inclusive**. `window(start, end)`, folds, slices: `end` **exclusive**.
- `phase_of` is 1-based. Absent covariate blocks are passed as zeros of shape `(T, 0)`.

### 3.8 Open points (record the decision in STATUS if you hit them)

- `PersistenceError` maps to exit code `1` in the CLI (no code in Part I overview §4.4); Part II overview §5.4 keeps
  the codes 0/1/2/3/4 in the core and 5 for `ProviderError` in the finance app.
- `changelog.d/` fragments: `changelog.d/<ID>.<feature|bugfix|doc|removal|misc>.md` for every user-visible change.

### 3.9 Part II decisions for this repository

- **Baseline.** The golden values of tag **`v0.2.0`** (E10-T01, `tests/golden/v0_2_0/`) are the reference for every
  Part II ticket here. Never regenerate them to make a test pass. Regenerating is allowed only after a toolchain
  change (JAX/NumPy/backend), with the owner's approval, recorded in the golden `manifest.json` and in STATUS.
- **Contracts.** The `PreparedDataset`/`ObservationSet` field sets stay frozen except where a Part II ticket
  changes them explicitly: `ObservationSet` is generalized in E10-T04a (`values`, `series_names`, `source`,
  `sequence_ids`); `PreparedDataset.clock` becomes a registered-name `str` in E10-T03.
- **Allow-lists.** `tests/fixtures/neutral_names_allowlist.txt` and `finance_import_allowlist.txt` (E10-T01) only
  shrink. Every entry carries the tag of the ticket that removes it. E10-T07b deletes both files.
- **Names stay `regimekit` here.** `RegimekitError`, `REGIMEKIT_TYPECHECK`, `REGIMEKIT_RECOVERY_T`, `.rkmodel`
  bundles and the `regimekit` CLI keep their names. The `hsmmkit` renames (Part II overview §2.3) happen only in
  the `hsmmkit` repository (E10-T08).
- **EM.** `estimation/em.py` (E5-T06) is moved, not rewritten, during M7a. E13-T02a in `hsmmkit` supersedes it.
- **Spikes.** `statespace/factorial.py` (E9-T02) goes to `hsmmkit` with the core; the MS-GARCH spike (E9-T03,
  `model/emission/msgarch.py`) moves to `regimekit.finance.emission` in E10-T06 and to `regimekit.emission.msgarch` in E10-T09.
- **Feature gates** (Part II overview §5.11) are introduced in `hsmmkit` (E10-T10). Do not add gated Part II
  features here.

## 4. Invariants (short form; details in the overviews)

1. Layer rule: `model/`, `statespace/`, `engine/`, `estimation/`, `inference/` import nothing from
   `regimekit.data` except `data.contracts` (and `data.batch` once it exists in `hsmmkit`). They contain no I/O and
   no logging inside jitted code.
2. Profile/dependency rule: no core package imports `regimekit.finance`. From E10-T01 on, `cli/` and `project.py` are
   no longer exempt; until E10-T07b their remaining imports are listed in the allow-list. Never weaken or
   special-case the AST tests.
3. Neutral-name rule (from E10-T01): no finance term in core modules beyond the allow-list (Part II overview §5.3).
4. The series dimension `M` is always present.
5. Structure lives in builders, never as special cases inside engines (D2). All parameters go through
   `ParamSpec`/`ParamManager`.
6. No NaN/Inf leaves a public function (→ `NumericalError`, checked outside jit). Errors carry `path=`.
7. No new runtime dependency without an ADR. Dev-group additions need a one-line reason in the commit body.
8. Guardrail during M7a (E10-T01…T07b): no new behaviour, no batch dimension `B`, no new families, no new extras.
   Those belong to `hsmmkit` tickets (E11–E16).

## 5. JAX gotchas

- `jnp.where` does not stop NaN gradients from the untaken branch. When masking `log(0)`, `−inf` or a division,
  sanitize the input too ("double where"). An all-`−inf` row in `logsumexp` gives NaN gradients — guard it.
- x64 must be on before any array exists: `tests/conftest.py` imports `regimekit` before anything creates arrays.
- `jax.devices("gpu")` **raises** when no GPU backend exists (e.g. `JAX_PLATFORMS=cpu`); catch it in `setup("auto")`
  and in the GPU fixture.
- Static structure (`ModelSpec`, `StateSpace`, `StateIndex`, `ParamManager`) is closed over or marked
  `eqx.field(static=True)`. NumPy arrays are unhashable: give such objects a stable `__hash__`/`__eq__` or keep the
  arrays as leaves. Verify "no recompilation" criteria with the jitted function's `_cache_size()`.
- Hypothesis plus JIT: the `tests/conftest.py` profile uses `deadline=None`. Do not lower `max_examples` to make
  tests pass.
- Bitwise golden tests: a refactor that reorders floating-point operations breaks them. Keep the operation order
  of the code you move (E10-T06: reuse the GBM helper instead of rewriting the Gaussian log-density).

## 6. Testing rules

- Test functions are prefixed with the ticket id in snake case: `test_e10_t02a_…`; files as listed under the
  ticket's "Creates".
- Test directories: the Part I overview list plus `tests/finance/`, `tests/live/` (every test `@pytest.mark.live`),
  `tests/spikes/`, and from Part II `tests/contract/` and `tests/golden/v0_2_0/`.
- `tests/reference/` is pure NumPy/SciPy and never imports `regimekit` numerics.
- Tolerances come from the overviews (Part I §4.7, Part II §5.7). **Never loosen a tolerance, add retries, shrink
  sample sizes, regenerate golden values or skip/xfail/delete a test to get green.**
- No network in default tests; providers take injectable downloaders/clocks. Tests never touch
  `~/.cache/regimekit` — use `tmp_path`.
- Fixtures are synthetic or trimmed to a few rows; raw provider data is never committed (NFR-08).
- NFR-02 timings refer to an 8-core x86 CPU. Record measured numbers and the hardware in STATUS; do not tune
  thresholds to the machine you run on (cloud VMs differ from the reference machine).

## 7. Definition of Done additions

Part I overview §4.8 and Part II overview §5.8 apply. In addition:
- Every acceptance criterion maps to a named test, or — for criteria that need CI, a GPU, network, another
  repository or the owner — is marked in STATUS with what was verified locally.
- M7a tickets: v0.2.0 golden test green, neutral-name scanner green, allow-list entries tagged with this ticket
  removed, `nox -s core` run and its result recorded in STATUS.
- When a ticket says it *replaces* or *moves* code, remove the old code in that ticket.
- A `changelog.d/` fragment exists for user-visible changes. `spec/STATUS.md` is updated.

## 8. Git workflow

- Branch per ticket from `main`: `feat/<ID>-<slug>` (e.g. `feat/E10-T02a-component-registries`). Commit freely on
  the branch.
- **Local sessions:** close with a squash merge on `main`, unless the owner asked for a pull request.
- **Cloud sessions:** push the ticket branch and open a pull request against `main`. The owner reviews and
  squash-merges. Never push to `main`.
- The squash commit message (or PR description) replaces the PR description of Part I overview §4.8:

  ```
  feat(config): E10-T02b Config extension, FinanceConfig, schema v2

  Requirements: FR-G05, FR-G07, D24
  Acceptance: AC1 → test_e10_t02b_appendix_a_v1_migrates; AC2 → …
  Golden: tests/golden/test_e10_t01_v0_2_0.py bitwise green
  Deviations: none | ADR-00NN | CLAUDE.md §3.x
  ```
- **Never tag, publish or create releases**, never delete branches, and never push to another repository, unless
  the owner asks for it in the current session (E10-T08 is the only planned cross-repository push).
- Never commit caches, `runs/`, `.env`, API keys or raw provider data.

## 9. Secrets and environment

- `FRED_API_KEY`, `STOOQ_API_KEY`, `TIINGO_API_KEY` come from the environment only: never hardcode, log or write
  them. In cloud sessions they are configured on the cloud environment, not in the repository.
- Other env vars: `REGIMEKIT_TYPECHECK`, `REGIMEKIT_RECOVERY_T` (base size, default 20000, `order2` runs at 2.5x),
  `HSMM_CORE_ONLY` (E10-T01: blocks `regimekit.finance` and the provider packages), `CI` (JSON logs),
  `JAX_PLATFORMS`.
- Linux/WSL2, CPU by default; nothing may require a GPU. `-m live` runs only when the user allows it.

## 10. Decide yourself

Private helper structure, internal names consistent with the overviews and §3, test organisation, order of work
inside a ticket, small refactors in modules the ticket touches, and choices a ticket explicitly leaves open
("choose …"). Record non-obvious choices in the ticket's STATUS notes.

## 11. Stop and mark BLOCKED (with a precise question and the evidence)

- Ticket and concept disagree and it is unclear which reading is right.
- An acceptance criterion looks unattainable (tolerance, performance, recovery rate).
- A new runtime dependency seems necessary beyond those the tickets name.
- A contract owned by another, already merged ticket would have to change in a way §3 doesn't cover.
- The spec is silent on something that affects public API, config schema or numerical results.
- The same failure survives three distinct fix attempts.
- **Part II:** a v0.2.0 golden value changes and the cause is not a documented toolchain change; an allow-list
  would have to grow; the ticket belongs to the `hsmmkit` repository; a step would push to another repository,
  create a tag or release, or rewrite history (E10-T08) without the owner's instruction in this session.

## 12. `spec/STATUS.md` format

```markdown
# Status
## Tickets
| ID | Title | Size | Phase | Deps | Status | Commit | Notes |
<!-- Status: todo | in-progress | done | blocked | needs-owner | other-repo -->

## Handoff              <!-- current ticket, branch, done so far, next step, open questions -->
## Blocked / needs owner
## Spec fixes           <!-- ticket corrections, §3 decisions applied, unowned deliverables created -->
## Findings for the owner
## Follow-ups           <!-- work noticed that belongs to later tickets -->
## Needs CI / network / GPU
## Measurements         <!-- NFR-02 timings with hardware; recovery pass rates -->
```

- Keep the Part I rows. The first Part II session (E10-T01) adds a section `## Tickets — Part II` with one row per
  E10–E17 ticket. Use the Part II overview index for ID, title, size and dependencies, the phase number from
  Part II overview §6, and status `todo` for this repository's tickets and `other-repo` for `hsmmkit` tickets.
- When an `hsmmkit` release appears, note its version and the tickets it contains under "Handoff", so that
  dependencies of this repository's post-split tickets can be checked.
