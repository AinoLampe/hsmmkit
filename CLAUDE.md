# CLAUDE.md — regimekit

regimekit is a JAX-based Hidden Semi-Markov Model framework for market-regime detection and probabilistic
return forecasting (Python 3.13, uv, JAX + Equinox + Optax, pydantic v2, typer). It is built ticket by ticket
from a written specification. Target: release 0.1.0 (Part I, 63 tickets, epics E0–E9).

The ticket overview is binding and imported here (conventions, build order, tolerances, DoD, amendments):

@spec/tickets/00-OVERVIEW-build-order-and-conventions.md

This file adds (a) the precedence rules, (b) **harmonization decisions that resolve naming and placement
conflicts between concept and tickets** (§3), and (c) the working rules for autonomous implementation.

## 1. Sources of truth and precedence

| Path | Role |
|---|---|
| `spec/tickets/<ID>-*.md` | The unit of work. Each ticket is self-contained (Context, Creates, Acceptance criteria, Tests). |
| `spec/tickets/00-OVERVIEW-…md` | Conventions (§4), build order (§5), dependency graph (§6), index (§7), Part II amendments (§8). |
| `spec/concept/regimekit_concept.md` | Concept **text v1.0**. "Concept v1.1" = this text + overview §8. Read only when a ticket is ambiguous. |
| `spec/concept/extensions.md` | Part II. **Not a ticket source before v0.1.0.** Consult only §15.7 when a ticket cites it. |
| `spec/STATUS.md` | Progress ledger you maintain (§12). |

Precedence, highest first:
1. **§3 of this file** (harmonization decisions). Where §3 contradicts a ticket or the concept, follow §3 and do not edit the ticket.
2. **Overview §8** amendments and guardrail.
3. **Concept** for semantics: formulas, defaults, acceptance numbers, tolerances, behaviour.
4. **Ticket** for everything the concept leaves open: exact signatures, file names, column names, test design.

If a ticket *contradicts* the concept on semantics (not a naming or detail refinement) and §3 doesn't cover it,
the overview rule applies: the concept wins and the ticket is fixed. Implement the concept's version, correct
the ticket file in a separate commit `docs(spec): align <ID> with concept §x.y`, and log it in STATUS →
"Spec fixes". If it is unclear which reading is right, stop (§11).

Never edit `spec/concept/*`. Never edit the overview. Ticket files only as described above.

Known slips in the concept text: §0 and §7 say "Section 6.5" / "6.5.4" for testing and recovery; correct is
**§6.4 / §6.4.4**. The concept still shows v1.0 names (e.g. `PreparedDataset.returns`) — §3.1 lists the v1.1 names.

## 2. Commands

Available once E0-T01/E0-T02 land; before that, those tickets create them.

```bash
uv sync --all-groups                                     # never pip
uv run nox -s lint types tests                           # full gate; must be green before merging a ticket
uv run pytest tests/<layer>/test_e4_t03a_*.py -x -q      # fast loop on the current ticket
uv run ruff format . && uv run ruff check --fix .
uv run mypy
uv run python scripts/check_package_coverage.py          # per-package ≥ 85 % (after a --cov run)
uv run nox -s docs                                       # mkdocs build --strict (after E0-T04)
uv run nox -s recovery                                   # -m slow (long)
uv run nox -s bench                                      # -m benchmark
```

- The nox `tests` session sets `REGIMEKIT_TYPECHECK=1` and `JAX_PLATFORMS=cpu`; the default pytest run
  excludes `slow`, `live`, `benchmark` and `gpu` (E0-T02 `addopts`).
- Add dependencies only with `uv add` / `uv add --group dev`. Never edit `uv.lock` by hand.

## 3. Harmonization decisions (override ticket and concept text)

### 3.1 v1.1 names — never reintroduce the v1.0 names

| v1.0 / conflicting text | Canonical |
|---|---|
| `PreparedDataset.returns` (concept §5.4, E9-T04 text) | `PreparedDataset.observations`. Config keys `data.preparation.returns` / `missing_returns` stay (Appendix A). Local variables for sampled log-returns may be called `returns`. |
| "`dt`: Δ_t in years" (E1-T01 comment) | `dt` is in **clock units**: years under `trading`/`calendar`, steps (≡ 1) under `steps`. Same for all parameters: annualized or per step, never both. |
| E3-T04 "Parameters (annualized)", init formulas with `/252` | Explicit duration families are in clock units too; init goes through `days_to_steps` (§3.2) and `init_from_mean_sojourn` (§3.3). |
| `mean_sojourn_years(theta)` | `mean_sojourn(theta, nominal_dt)` |
| `estimation/fit.py` | `estimation/map.py`; `from regimekit.estimation import fit` |
| `data/providers/` | `finance/providers/` |
| `model/params.py` | package `model/params/` (`bijectors`, `priors`, `spec`, `manager`; `__init__` re-exports) |
| `@jaxtyped(typechecker=beartype)` | `from regimekit.utils.typing import typechecked` (active only with `REGIMEKIT_TYPECHECK=1`) |
| `Estimator.fit(spec, data, cfg, key)` | `fit(spec, data, cfg, seed: int) -> FitResult` |
| `jump_probs(theta, z_pi, memory_index: StateIndex)` | `jump_probs(theta, z_pi, memory: MemoryLayout)`; `MemoryLayout` lives in **`model/transition.py`** (E3-T05 left `covariates.py` open). E1-T05's protocol stub declares the `MemoryLayout` dataclass fields so mypy passes early. |
| `StateIndex.memory_tuple_index()` (E1-T05) vs `row_of_state` (E4-T02a/b) | Array field **`row_of_state (S,)`**; no method. All other index arrays as placed by E4-T02a/b. |
| `InferenceEngine.loglik -> float` | Engine methods: `loglik`, `forward`, `forward_backward`, `viterbi`, `approx`. `loglik` and `Posterior.loglik` are 0-d `Array`s (pytree leaves). `PosteriorResult.loglik` and `FitResult.loglik` are Python `float`. |
| `Posterior.elapsed_posterior()` "dispatches on engine tag" | `Posterior` has a static field `engine: Literal["expanded", "explicit"] = "expanded"` (added by E4-T04, set by E4-T06a). |
| `PosteriorResult.viterbi` | The **regime** path `(T,)`; expanded paths stay inside `augmented`. |
| `data.irregular` (E4-T06a) | No such field — the `PreparedDataset` field set is frozen. Use `utils.clock.is_irregular(dt)` (`max/min > 1.5`). |

### 3.2 Placement decisions (keep the layer and profile rules of overview §4.3 intact)

| Conflict | Decision |
|---|---|
| E6-T04a puts `future_grid` in `data/calendar.py`, but `inference/` may not import `data.calendar`. | New module **`regimekit/utils/clock.py`** holds `TRADING_DAYS_PER_YEAR = 252`, `CALENDAR_DAYS_PER_YEAR = 365.25`, `future_grid(...)`, `days_to_steps(days, clock, nominal_dt, bars_per_day)` and `is_irregular(dt)`. `data/calendar.py` imports from it. |
| E3-T03 says "no literal 252 outside `calendar.py`"; E3-T04 uses `/252`; conversion needs the clock name. | No literal `252`/`365.25` outside `utils/clock.py`. Branching on the clock *name* is allowed only in `utils/clock.py` and `data/calendar.py`. |
| E2-T05 `CachedProvider(inner: DataProvider)` would make `data/` import `regimekit.finance` (profile rule). | `data/cache.py` declares its own structural Protocol `CacheableProvider` (`fetch_prices`, `fetch_series`); finance providers satisfy it structurally. |
| E0-T01 `.gitignore` has `*.parquet`; E2-T02/E8-T03 commit parquet fixtures. | `.gitignore` adds `!tests/fixtures/**/*.parquet`. |
| E5-T05b moves the recovery harness from `tests/recovery/harness.py` into `src/`. | E5-T05a creates it directly as **`src/regimekit/estimation/recovery.py`** (pure, no pytest, no file I/O). Give it small default-suite tests so `estimation` coverage stays ≥ 85 %. |

### 3.3 Unowned deliverables and protocol completions

- **Config → model assembly (no ticket owns it).** `src/regimekit/model/build.py`:
  `build_model_spec(cfg: ProjectConfig, data: PreparedDataset) -> ModelSpec`, plus name registries
  `EMISSION_FAMILIES`, `DURATION_FAMILIES` keyed by the config names. Every family gets a `from_config(...)`
  classmethod (the E3-T01 pattern). `init_mean_sojourn_days` is converted here with `days_to_steps`. The first
  ticket that needs a `ModelSpec` from a config creates it (at the latest E6-T02); E8-T01 reuses the registries.
  Log it in STATUS → "Spec fixes".
- **`DurationFamily`** additionally has `supports_explicit: bool`,
  `init_from_mean_sojourn(mean_steps, nominal_dt) -> Params` (referenced by E5-T03) and the per-step wrapper
  `advance_prob_t` (E4-T02b). **`TransitionModel`** gets `jump_probs_t`. Phase families size their rate arrays
  by `n_rates` (`N`, or `N(N−1)` for `pair` conditioning, E4-T02a).
- **Class names:** `GBMFamily`, `StudentTFamily`; `GeometricDuration`, `NegBinDuration`, `PoissonDuration`,
  `GammaDuration`, `WeibullDuration`, `LogNormalDuration`, `NonparametricDuration`; `JumpTransition`;
  `ExpandedHMMEngine`, `ExplicitDurationEngine`; `MonteCarloForecaster`. Registry keys = config names
  (`gbm`, `student_t`, `geometric`, `negbin`, …).
- **ParamSpec names** are `<component>.<config key>`:
  `emission.{mu,sigma,gamma,delta}` (GBM), `emission.{m,s,nu,gamma,delta}` (Student-t),
  `duration.{rate,c}` (phase), `duration.{kappa,shape,scale,mu_log,sd_log,logits,c}` (explicit),
  `transition.{a,b}` or `transition.pi_rows` (no covariates), `initial.logits`.

### 3.4 Config fields introduced after E1-T02a

Added by the ticket that uses them, with defaults that keep Appendix A valid and warning-free; update the
JSON-Schema test and the resolved-dump test in the same ticket.

| Field | Ticket |
|---|---|
| `estimation.chunk_size: int \| None = None` | E4-T06b |
| `estimation.parallel_scan: bool = False` | E4-T07 |
| `forecast.covariate_forecaster`: `bootstrap.mean_block_length`, `ar.diagonal = False`, `scenario.path` (check whether E1-T02a already has them) | E6-T03 |
| `evaluation.backtest.n_sim: int \| None = None` | E7-T02b |
| `evaluation.strategy.cap: float = 1.0`, `evaluation.strategy.allow_short: bool = False` | E7-T03 |
| `model.emission.per_series` | E9-T01 (post-0.1.0) |

### 3.5 RNG

- `key_for(seed, *ids)` takes **ints only**. Purpose ids are named int constants in `utils/rng.py`
  (E6-T04b's `"forecast"` becomes e.g. `PURPOSE_FORECAST`).
- `MonteCarloForecaster.forecast(..., seed: int)`. E7-T02b needs a per-origin seed: use
  `utils.rng.derive_seed(seed, *ids) -> int` (deterministic), not a key.

### 3.6 ADR numbers (the E9 tickets collide with the amendment ADRs)

| ADR | Content | Written by |
|---|---|---|
| 0001–0010 | Concept §6.7 baseline | E0-T04 |
| 0011 | safetensors bundle format | E8-T01 |
| 0012 / 0013 | core/profile split / `steps` clock | E0-T04 |
| 0014–0021 | reserved for Part II | — |
| **0022** | in-scan emissions, annualization exception (E9-T03 text says 0012) | E9-T03 |
| **0023** | endogenous covariates (E9-T04 text says 0013) | E9-T04 |
| 0024+ | ADRs raised during implementation | any |

### 3.7 Index and range conventions

- Code index `t` (0-based) = math step `t+1`. `A_t` moves from code index `t−1` to `t`; covariate row `t` is the
  lagged value usable at step `t` (known at the end of `t−1`).
- Segments `(start, end, regime)`: `end` **inclusive**. `window(start, end)`, folds, slices: `end` **exclusive**.
- `phase_of` is 1-based. Absent covariate blocks are passed as zeros of shape `(T, 0)`.

### 3.8 Open points (record the decision in STATUS if you hit them)

- `PersistenceError` has no exit code in overview §4.4 while E8-T02 maps every `RegimekitError`: map it to `1`
  and log a finding for the owner.
- `changelog.d/` fragments (overview §4.9) are written from the first user-visible change as
  `changelog.d/<ID>.<feature|bugfix|doc|removal|misc>.md`; the towncrier config arrives with E8-T04.

## 4. Invariants (short form; details in overview §4.3–§4.6)

1. Layer rule: `model/`, `statespace/`, `engine/`, `estimation/`, `inference/` import nothing from
   `regimekit.data` except `data.contracts`. They contain no I/O, no logging inside jitted code.
2. Profile rule: no core package imports `regimekit.finance`; only `cli/` and `project.py` may.
   Both rules are enforced by the AST test from E1-T01 — never weaken or special-case it.
3. The series dimension `M` is always present.
4. Structure lives in builders, never as special cases inside engines (D2). All parameters go through
   `ParamSpec`/`ParamManager`.
5. No NaN/Inf leaves a public function (→ `NumericalError`, checked outside jit). Errors carry `path=`.
6. No new runtime dependency without an ADR. Dev-group additions (e.g. matplotlib/nbclient in E8-T03,
   towncrier/semantic-release in E8-T04) need a one-line reason in the commit body.
7. Guardrail (overview §8): no batch dimension `B`, no `profile: core` execution path, no new extras in 0.1.0.

## 5. JAX gotchas (in addition to overview §4.5)

- `jnp.where` does not stop NaN gradients from the untaken branch. When masking `log(0)`, `−inf` or a division,
  sanitize the input too ("double where"). An all-`−inf` row in `logsumexp` gives NaN gradients — guard it.
- x64 must be on before any array exists: `tests/conftest.py` imports `regimekit` before anything creates arrays.
- `jax.devices("gpu")` **raises** when no GPU backend exists (e.g. `JAX_PLATFORMS=cpu`); catch it in `setup("auto")`
  and in the GPU fixture.
- Static structure (`ModelSpec`, `StateSpace`, `StateIndex`, `ParamManager`) is closed over or marked
  `eqx.field(static=True)`. NumPy arrays are unhashable: give such objects a stable `__hash__`/`__eq__` or keep the
  arrays as leaves. Verify "no recompilation" acceptance criteria with the jitted function's `_cache_size()`.
- Hypothesis plus JIT: register a profile with `deadline=None` in `tests/conftest.py` (first-call compilation
  exceeds the default deadline). Do not lower `max_examples` to make tests pass.

## 6. Testing rules (in addition to overview §4.7)

- Test functions are prefixed with the ticket id in snake case: `test_e4_t03a_…`; files as listed under the
  ticket's "Creates".
- Test directories: the overview list plus `tests/finance/` (shared provider contract helper, not collected),
  `tests/live/` (every test `@pytest.mark.live`) and `tests/spikes/` (E9).
- `tests/reference/` is pure NumPy/SciPy and never imports `regimekit` numerics; otherwise golden tests are circular.
- Tolerances come from overview §4.7 / concept Appendix D. **Never loosen a tolerance, add retries, shrink sample
  sizes or skip/xfail/delete a test to get green.** The only sanctioned xfail is E5-T01's explicit-engine
  parametrization, and it is avoided by doing E4-T06a before E5-T01.
- No network in default tests; providers take injectable downloaders/clocks. Tests never touch
  `~/.cache/regimekit` — use `tmp_path`.
- Fixtures are synthetic or trimmed to a few rows; raw provider data is never committed (NFR-08).
- NFR-02 timings refer to an 8-core x86 CPU. Record measured numbers and the hardware in STATUS; do not tune
  thresholds to this machine.

## 7. Definition of Done additions

Overview §4.8 applies. In addition:
- Every acceptance criterion maps to a named test, or — for criteria that need CI, a GPU, network or the owner —
  is marked in STATUS with what was verified locally.
- Amended tickets (banner "Amended 2026-09-14") list "(Part II amendment 15.7)" in the acceptance mapping.
- When a ticket says it *replaces* a stub or temporary helper (E1-T05 protocol stubs, E6-T02 temporary loader),
  remove the old code in that ticket.
- A `changelog.d/` fragment exists for user-visible changes. `spec/STATUS.md` is updated.

## 8. Git workflow

- Branch per ticket from `main`: `feat/<ID>-<slug>` (e.g. `feat/E4-T03a-expanded-forward`). Commit freely on the branch.
- Close with a squash merge on `main`. The commit message replaces the PR description of overview §4.8:

  ```
  feat(engine): E4-T03a Expanded engine: forward / loglik

  Requirements: FR-I01, NFR-01, NFR-02
  Acceptance: AC1 → test_e4_t03a_forward_matches_reference; AC2 → …; AC5 → benchmark (measured 41 ms)
  Deviations: none | ADR-00NN | CLAUDE.md §3.x
  ```
- **Never push, tag, publish or create releases** unless the user asks in the current session. E8-T04 stops
  before `git tag v0.1.0`.
- Never commit caches, `runs/`, `.env`, API keys or raw provider data.

## 9. Secrets and environment

- `FRED_API_KEY`, `STOOQ_API_KEY`, `TIINGO_API_KEY` come from the environment only: never hardcode, log or write them.
- Other env vars: `REGIMEKIT_TYPECHECK`, `REGIMEKIT_RECOVERY_T` (E5-T05a), `CI` (JSON logs), `JAX_PLATFORMS`.
- Linux/WSL2, CPU by default; nothing may require a GPU. `-m live` runs only when the user allows it.

## 10. Decide yourself

Private helper structure, internal names consistent with overview §4.6 and §3 above, test organisation, order of
work inside a ticket, small refactors in modules the ticket touches, and choices a ticket explicitly leaves open
("choose …"). Record non-obvious choices in the ticket's STATUS notes.

## 11. Stop and mark BLOCKED (with a precise question and the evidence)

- Ticket and concept disagree and it is unclear which reading is right.
- An acceptance criterion looks unattainable (tolerance, performance, recovery rate).
- A new runtime dependency seems necessary beyond those the tickets name (safetensors in E8-T01).
- A contract owned by another, already merged ticket would have to change in a way §3 doesn't cover.
- The spec is silent on something that affects public API, config schema or numerical results.
- The same failure survives three distinct fix attempts.

## 12. `spec/STATUS.md` format

```markdown
# Status
## Tickets
| ID | Title | Size | Phase | Deps | Status | Commit | Notes |
<!-- Status: todo | in-progress | done | blocked | needs-owner -->

## Handoff              <!-- current ticket, branch, done so far, next step, open questions -->
## Blocked / needs owner
## Spec fixes           <!-- ticket corrections, §3 decisions applied, unowned deliverables created -->
## Findings for the owner
## Follow-ups           <!-- work noticed that belongs to later tickets -->
## Needs CI / network / GPU
## Measurements         <!-- NFR-02 timings with hardware; recovery pass rates -->
```
