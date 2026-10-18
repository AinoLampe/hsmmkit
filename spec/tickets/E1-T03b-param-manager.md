# E1-T03b — `ParamSpec` and `ParamManager` (modes free / fixed / tied / bounded / prior, ordering constraints)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M03, FR-M05, FR-M11, D4 | M0 | §4.6, §4.10 (`p` count) |

## Goal
Map between the flat unconstrained optimizer vector `u ∈ ℝ^p` and the constrained parameter pytree `θ` according to per-parameter, per-state modes.

## Context (self-contained)
```python
@dataclass(frozen=True)
class ParamSpec:
    name: str                       # e.g. "emission.sigma"
    shape: tuple[int, ...]          # e.g. (N,) or (N, K)
    domain: str                     # real | positive | interval | simplex_ref | nu_gt_2 | pmf
    mode: str | tuple[str, ...]     # one mode, or one per leading index (per state)
    value: np.ndarray | None        # required for fixed entries
    init: np.ndarray | None         # initial θ for free entries
    bounds: tuple[float, float] | None
    prior: Prior | None
    tie_group: str | None           # entries sharing a group share ONE u entry
    constraint: Literal["none", "increasing"] = "none"

class ParamManager:
    def __init__(self, specs: tuple[ParamSpec, ...]): ...
    n_free: int                                          # p, counts entries of u
    def unflatten(self, u: Array) -> Params: ...         # u → θ (dict name → array, full shape)
    def flatten(self, theta: Params) -> Array: ...       # θ → u (only free coordinates)
    def log_det_jac(self, u: Array) -> Array: ...        # Σ over free coordinates
    def log_prior(self, theta: Params) -> Array: ...     # Σ over params with mode prior (or a prior set)
    def init_u(self) -> Array: ...                       # flatten(init values)
    def names(self) -> list[str]: ...                    # one label per u coordinate, e.g. "emission.sigma[1]"
```
Rules: `fixed` ⇒ no entry in `u`, `θ` takes `value`. `tied` ⇒ one `u` entry per `tie_group`, broadcast. `bounded` ⇒ `interval` bijector with `bounds`. `prior` ⇒ free + prior term. `free` with `domain=positive` uses softplus. `constraint=increasing` applies the `increasing` bijector over the state axis of a whole spec (cannot be combined with per-state `fixed` on that spec; raise `ConfigError`). Modes given per state produce a mixed layout: the free entries of the array are gathered into `u` in row-major order. `Params` is a `dict[str, Array]` (pytree). `ParamManager` itself is static (hashable) so it can be closed over by jitted functions.

## Creates
- `src/regimekit/model/params/manager.py`, `src/regimekit/model/params/spec.py`
- `tests/property/test_e1_t03b_param_manager.py`
- `docs/math/parameters.md` (modes section)

## Acceptance criteria
1. Round trip `unflatten(flatten(θ)) == θ` for random valid `θ` (hypothesis over spec layouts mixing all five modes) ≤ 1e-9; `flatten(unflatten(u)) == u` ≤ 1e-9.
2. `fixed` entries are absent from `u` and unchanged after `unflatten`; `tied` entries share exactly one coordinate (changing it changes all group members).
3. `n_free` equals the hand-computed count for Appendix B cases (`N=4`, GBM+NB order 1 → 20 with a full jump matrix, no covariates).
4. `log_det_jac` and `log_prior` are jittable and differentiable; `log_prior` of a spec without priors is exactly 0.
5. `constraint=increasing` produces strictly increasing `θ` for any `u`.

## Tests
Property tests; Appendix B parameter counts; error cases (`fixed` without `value`, `increasing` + per-state `fixed`).

## Out of scope
Family-specific specs (E3-T*); objective assembly (E5-T01).

## Connections
- Depends on: E1-T03a
- Blocks: E1-T05, E3-T01, E3-T03, E3-T05, E5-T01
- Relates to: E3-T06 (lattice generates `fixed` specs), E4-T05 (`p` for AIC/BIC)
