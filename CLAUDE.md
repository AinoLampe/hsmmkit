# CLAUDE.md — hsmmkit (Python package `hsmmkit`)

**`hsmmkit`** is a domain-neutral JAX framework for Hidden Semi-Markov Models (Python 3.13, uv, JAX + Equinox +
Optax, pydantic v2, typer). It was extracted with its history from the repository `RegimeDetectionAndForecast`
(package `regimekit`, the finance application) by ticket **E10-T08**, and is extended ticket by ticket from the
written specification.

- **Part I** (E0–E9) built the core inside `RegimeDetectionAndForecast` (`regimekit` 0.1.0 / 0.2.0). Its tickets
  stay in `spec/tickets/` as history.
- **Part II** (E10–E17, 66 tickets) splits the core out and extends it. E10-T01…T07b (decouple) and E10-T08
  (this extraction) are done. The remaining `hsmmkit` tickets are the current work.

Both ticket overviews are binding and imported here. Part I conventions stay in force unless the Part II overview
changes them.

@spec/tickets/00-OVERVIEW-build-order-and-conventions.md
@spec/tickets/10-OVERVIEW-part2-build-order-and-conventions.md

This file adds (a) the scope rules for this repository (§0), (b) the precedence rules (§1), (c) **harmonization
decisions** (§3) and (d) the working rules for autonomous implementation.

## 0. Scope in this repository

Every Part II ticket names its repository in the `Repo` column (Part II overview §2.1).

| Repo value | Tickets | Where |
|---|---|---|
| `hsmmkit` | E10-T10, E10-T11, E11-T01…T06, E11-T08, E12-*, E13-*, E14-*, E15-*, E16-T01…T05, E17-T01…T04 | **here** |
| `RegimeDetectionAndForecast` | E10-T09, E11-T07, E11-T09, E16-T06, E17-T05 | **not here**. If asked to implement one, stop and say it belongs to the `RegimeDetectionAndForecast` repository. |

- **Picking the next ticket.** Follow the Part II build order (overview §6) and dependency graph (§7). The next
  ticket is the first `todo` row in `spec/STATUS.md` whose dependencies are `done`.
- **Downstream.** `RegimeDetectionAndForecast` (package `regimekit`) depends on `hsmmkit` with a minor-range pin.
  The nightly `downstream` job runs its suite against this repository's HEAD. A change that breaks it is either
  fixed here or comes with a matching `RegimeDetectionAndForecast` ticket (named under *Connections*).
- **Names.** `HsmmkitError`, `HSMMKIT_TYPECHECK`, `HSMMKIT_RECOVERY_T`, bundle suffix `.hsmm`, console script
  `hsmmkit`. The word `regimekit` appears only as the "example extension" in the guide.
- **Feature gates** (Part II overview §5.11) are introduced by E10-T10; a ticket removes its own gate.

## 1. Sources of truth and precedence

| Path | Role |
|---|---|
| `spec/tickets/<ID>-*.md` | The unit of work. Each ticket is self-contained (Context, Creates, Acceptance criteria, Tests). |
| `spec/tickets/00-OVERVIEW-…md` | Part I conventions (§4), build order, index, amendments. |
| `spec/tickets/10-OVERVIEW-part2-…md` | Part II: repositories and phases (§2), conventions and tolerances (§5), feature gates (§5.11), build order (§6), dependency graph (§7), index (§8), guardrails (§10). |
| `spec/concept/extensions.md` | Part II concept **v1.2**: the concept for every E10–E17 ticket. |
| `spec/concept/regimekit_concept.md` | Part I concept text v1.0 (+ Part I overview §8 = "concept v1.1"). Read only when a ticket is ambiguous. |
| `spec/STATUS.md` | Progress ledger you maintain (§12). |

Precedence, highest first: (1) §3 of this file, (2) the Part II overview (§5, §10), (3) `extensions.md` v1.2 for
Part II semantics and the Part I concept for every Part I semantic Part II does not change, (4) the ticket for
everything the concepts leave open.

If a ticket *contradicts* the concept on semantics (not a naming or detail refinement) and §3 doesn't cover it,
the concept wins and the ticket is fixed: implement the concept's version, correct the ticket file in a separate
commit `docs(spec): align <ID> with concept §x.y`, and log it in STATUS → "Spec fixes". If it is unclear which
reading is right, stop (§11). Never edit `spec/concept/*` or the overviews.

Known slips in the Part I concept text: §0 and §7 say "Section 6.5" / "6.5.4" for testing and recovery; correct
is **§6.4 / §6.4.4**.

## 2. Commands

```bash
uv sync --all-groups                                     # never pip
uv run nox -s lint types tests                           # full gate; must be green before merging a ticket
uv run pytest tests/<layer>/test_e10_t10_*.py -x -q      # fast loop on the current ticket
uv run ruff format . && uv run ruff check --fix .
uv run mypy
uv run python scripts/check_package_coverage.py          # per-package ≥ 85 % (after a --cov run)
uv run nox -s docs                                       # mkdocs build --strict
uv run nox -s recovery_smoke                             # nightly smoke: 1 scenario at T=5000
uv run nox -s recovery                                   # -m slow, full suite (weekly, hours)
uv run nox -s bench                                      # -m benchmark
uv run pytest tests/unit/test_e10_t01_neutral_names.py -q
uv run pytest tests/unit/test_e10_t08_package.py -q
```

- The nox `tests` session sets `HSMMKIT_TYPECHECK=1` and `JAX_PLATFORMS=cpu`. The default pytest run excludes
  `slow`, `live`, `benchmark` and `gpu` (E0-T02 `addopts`).
- Add dependencies only with `uv add` / `uv add --group dev`. Never edit `uv.lock` by hand.

## 3. Harmonization decisions (override ticket and concept text)

### 3.1 v1.1 names — never reintroduce the v1.0 names

| v1.0 / conflicting text | Canonical |
|---|---|
| `PreparedDataset.returns` (concept §5.4, E9-T04 text) | `PreparedDataset.observations`. Config keys `data.preparation.returns` / `missing_returns` stay in `schema_version: 1` (Appendix A); E10-T02b migrates them to v2 names. Local variables for sampled log-returns may be called `returns`. |
| "`dt`: Δ_t in years" (E1-T01 comment) | `dt` is in **clock units**: years under `trading`/`calendar`, steps (≡ 1) under `steps`. Same for all parameters: annualized or per step, never both. |
| E3-T04 "Parameters (annualized)", init formulas with `/252` | Explicit duration families are in clock units too; init goes through `days_to_steps` (§3.2) and `init_from_mean_sojourn` (§3.3). |
| `mean_sojourn_years(theta)` | `mean_sojourn(theta, nominal_dt)` |
| `estimation/fit.py` | `estimation/map.py`; `from hsmmkit.estimation import fit` |
| `model/params.py` | package `model/params/` (`bijectors`, `priors`, `spec`, `manager`; `__init__` re-exports) |
| `@jaxtyped(typechecker=beartype)` | `from hsmmkit.utils.typing import typechecked` (active only with `HSMMKIT_TYPECHECK=1`) |
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
| E6-T04a puts `future_grid` in `data/calendar.py`, but `inference/` may not import `data.calendar`. | **`hsmmkit/utils/clock.py`** holds `CALENDAR_DAYS_PER_YEAR = 365.25`, `future_grid(...)`, `days_to_steps(...)` and `is_irregular(dt)`, dispatching through the clock registry (kind `clock`; an application registers its own clocks). `data/calendar.py` imports from it. |
| Literal `252` / `365.25`. | No literal `252` anywhere in the core; `365.25` only in `utils/clock.py`. Branching on the clock *name* is allowed only in `utils/clock.py`, `data/calendar.py` and the clock registry. |
| E0-T01 `.gitignore` has `*.parquet`; fixtures commit parquet files. | `.gitignore` adds `!tests/fixtures/**/*.parquet`. |
| Recovery harness location. | **`src/hsmmkit/estimation/recovery.py`** (pure, no pytest, no file I/O). Its grid clock is `recovery_clock()`: the registered year clock with the fewest days per year. |

### 3.3 Unowned deliverables and protocol completions

- **Config → model assembly.** `src/hsmmkit/model/build.py`: `build_model_spec(cfg, data) -> ModelSpec`, with
  name registries `EMISSION_FAMILIES`, `DURATION_FAMILIES`. **From E10-T02a on** these dicts are thin views over
  `hsmmkit.plugins.REGISTRY`; new lookups go through the registry only.
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

Part II: the schema is the neutral `ProjectConfig` (`schema_version: 1`; applications subclass it and register their own migrations). A Part II ticket adds a config field to `ProjectConfig`; each addition updates the JSON-Schema test and the resolved-dump test in the same ticket.

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
| 0015 | Clock registry; the `trading` clock lives in the `hsmmkit` application | E10-T03 |
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
  the codes 0/1/2/3/4 in the core (`hsmmkit.cli.app.EXIT_CODES`); applications register their own.
- `changelog.d/` fragments: `changelog.d/<ID>.<feature|bugfix|doc|removal|misc>.md` for every user-visible change.

### 3.9 Decisions for this repository

- **Baseline.** `hsmmkit` 0.1.0 was extracted from `RegimeDetectionAndForecast` at its E10-T07b merge commit
  (`42528543b9f0a4b7856e72c809016aee5b12a3e9`). The core golden and reference tests (`tests/golden`,
  `tests/reference`) must pass unchanged; never regenerate golden values to make a test pass (regeneration needs a
  documented toolchain change and the owner's approval, recorded in STATUS).
- **Contracts.** The `PreparedDataset` / `ObservationSet` field sets stay frozen except where a ticket changes
  them explicitly.
- **Neutral names.** No allow-list: `tests/unit/test_e10_t01_neutral_names.py` finds no finance term in the core
  and no import of an application package. Never weaken or special-case it.
- **Extensions.** Applications register components (registries), subclass config sections and `Project`, and build
  their CLI with `build_app`; they never patch core objects (D24).
- **Feature gates** (Part II overview §5.11) are introduced by E10-T10; each ticket removes its own gate.
- **Tests that were dropped in the extraction.** About 50 test modules of `RegimeDetectionAndForecast` needed the
  finance package (Appendix A configs, providers, strategies) and were not carried over; the core-relevant
  scenarios they covered (fits, backtests, bundles, CLI on non-finance configs) are re-created with `gaussian`
  fixtures by the tickets that touch those areas (see STATUS → Follow-ups).

## 4. Invariants (short form; details in the overviews)

1. Layer rule: `model/`, `statespace/`, `engine/`, `estimation/`, `inference/` import nothing from
   `hsmmkit.data` except `data.contracts` (and `data.batch` once it exists in `hsmmkit`). They contain no I/O and
   no logging inside jitted code.
2. Dependency rule: no core package imports an application package (`hsmmkit` or any other); `hsmmkit` never imports `hsmmkit`. `cli/`, `project.py` and `__init__.py` are in scope. Never weaken or special-case the AST tests.
3. Neutral-name rule: no finance term in core modules (Part II overview §5.3); there is no allow-list.
4. The series dimension `M` is always present.
5. Structure lives in builders, never as special cases inside engines (D2). All parameters go through
   `ParamSpec`/`ParamManager`.
6. No NaN/Inf leaves a public function (→ `NumericalError`, checked outside jit). Errors carry `path=`.
7. No new runtime dependency without an ADR. Dev-group additions need a one-line reason in the commit body.
8. Feature gates: a key whose implementation has not landed is rejected at model build (E10-T10).

## 5. JAX gotchas

- `jnp.where` does not stop NaN gradients from the untaken branch. When masking `log(0)`, `−inf` or a division,
  sanitize the input too ("double where"). An all-`−inf` row in `logsumexp` gives NaN gradients — guard it.
- x64 must be on before any array exists: `tests/conftest.py` imports `hsmmkit` before anything creates arrays.
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
- Test directories: the Part I overview list plus `tests/live/` (every test `@pytest.mark.live`),
  `tests/spikes/`, and from Part II `tests/contract/` and `tests/golden/v0_2_0/`.
- `tests/reference/` is pure NumPy/SciPy and never imports `hsmmkit` numerics.
- Tolerances come from the overviews (Part I §4.7, Part II §5.7). **Never loosen a tolerance, add retries, shrink
  sample sizes, regenerate golden values or skip/xfail/delete a test to get green.**
- No network in default tests; providers take injectable downloaders/clocks. Tests never touch
  `~/.cache` — use `tmp_path`.
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
  feat(estimation): E13-T02a EMEstimator: closed-form M-steps

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
- Other env vars: `HSMMKIT_TYPECHECK`, `HSMMKIT_RECOVERY_T` (base size, default 20000, `order2` runs at 2.5x),
  `CI` (JSON logs),
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
