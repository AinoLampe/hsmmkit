# E11-T01 — `family_contract` test-suite in `hsmmkit.testing`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EM08, FR-PL02, FR-EM07 | M8 | Part II §14.2 (extended family contract), §15.3, §16.2, §16.4 |

## Goal
Publish one importable pytest suite that every emission family — built-in or external — must pass, and extend the `EmissionFamily` protocol with the optional estimation hooks.

## Context (self-contained)
Protocol extension (`hsmmkit.model.emission.base`), all new methods optional with default `None`-returning implementations in a `BaseEmission` mixin:
```python
class EmissionFamily(Protocol):
    name: str; native_time: Literal["continuous", "discrete"]
    support: Literal["real", "count", "binary", "categorical"]; scale_by_dt: bool
    def param_specs(self, n_states, n_series, cov_names) -> tuple[ParamSpec, ...]: ...
    def log_prob(self, y, dt, theta, z) -> Array: ...                 # (T, N)
    def sample(self, key, regime, dt, theta, z) -> Array: ...         # (T, M)
    def init_carry(self) -> Any: ...; def step_carry(self, carry, y_t, theta) -> Any: ...
    def sufficient_stats(self, y, dt, gamma, z) -> Any | None: ...    # gamma (T, N) regime weights
    def m_step(self, stats: Any) -> Params | None: ...
    def conjugate_prior(self) -> PriorFamily | None: ...
    def expected_log_prob(self, y, dt, q_theta, z) -> Array | None: ... # VB
    def moments(self, regime, dt, theta, z) -> tuple[Array, Array] | None: ...   # mean, var (T, M) if analytic
```
Suite (`hsmmkit/testing/family_contract.py`), used as:
```python
from hsmmkit.testing import family_contract
family_contract.register_case(FamilyCase(name="gaussian-diag", factory=..., theta_factory=..., data_factory=..., scipy_logpdf=...))
pytest_plugins = ["hsmmkit.testing.family_contract"]   # yields test_family_contract[<case>-<check>]
```
`FamilyCase`: `factory() -> family`, `theta_factory(key, N, M, K) -> θ`, `data_factory(key, T, M) -> (y, dt, z)` respecting the support, optional `scipy_logpdf(y, dt, θ, z) -> (T, N)`, `clocks: tuple[str,...] = ("steps", "calendar")` (extensions add theirs, e.g. `trading`), `tolerance` overrides only with justification.
Checks (each a separate parametrized test):
1. `log_prob` shape `(T, N)`, finite on valid data; vs `scipy_logpdf` ≤ 1e-10 when given.
2. `sample` support-valid; empirical mean/var within 3 SE of `moments()` (or of Monte-Carlo from `log_prob` via importance check when `moments` is `None`), `n = 20 000` per state.
3. Δ-scaling: if `scale_by_dt`, moments scale with `Δ` as declared (`mean ∝ Δ`, `var ∝ Δ` for location-scale families); else `log_prob` independent of `dt`.
4. Support validation: preparer rejects out-of-support data for this family (E10-T05 path).
5. Covariate links: zero coefficients reproduce the covariate-free density bitwise; non-zero coefficients change it.
6. Param-spec round trip: `ParamManager` `θ = θ(u(θ))` ≤ 1e-12; spec names unique; jit + grad finite.
7. Optional-method consistency: if `sufficient_stats`/`m_step` exist, `m_step` equals the L-BFGS optimum of `Σ_t Σ_i γ_ti log f_i(y_t)` ≤ 1e-8 relative (Appendix G); if `conjugate_prior` exists, the posterior update with zero data returns the prior.
8. Every registered clock in `case.clocks` evaluates without branching on clock names (`dt` arrays from each clock's `dt()`).
Built-in cases registered now: `gaussian` (diag, with and without GBM options), `student_t` (uv), `product` (E9-T01).

## Creates
- `src/hsmmkit/model/emission/base.py` (protocol + `BaseEmission`), `src/hsmmkit/testing/family_contract.py`, `tests/contract/test_e11_t01_builtin_families.py`
- `docs/guide/extending/families.md` (how to use the suite from another package)

## Acceptance criteria
1. `gaussian`, `student_t`, `product` pass all applicable checks.
2. A deliberately broken family fixture (wrong normalizing constant) fails check 1 or 2 with a readable message.
3. The suite is importable from an external package (test creates a temp package that registers a case and runs it via `pytest --pyargs`).
4. Appendix G tolerances are the only tolerances used.

## Tests
Meta-tests of the suite on good and broken fixtures; external-package usage test.

## Out of scope
Entry-point discovery (E16-T04); new families (E11-T02…T08).

## Connections
- Depends on: E10-T08
- Blocks: E11-T02, E11-T04, E16-T04
- Relates to: E10-T03 (clock contract), E10-T04a (source contract), E10-T09 (`regimekit` uses the suites)
