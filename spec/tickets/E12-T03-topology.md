# E12-T03 — Transition topology `Ω`: shorthands, masked logits, absorbing states in both engines

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-DU01, FR-DU02, D16 | M9 | Part II §14.5, Appendix F (parameter counts); Part I §4.4, E3-T05, E4-T02b |

## Goal
Allow structural zeros in the jump matrix (left-to-right, banded, cyclic, arbitrary) and absorbing states, with correct parameter counts and behaviour in both engines.

## Context (self-contained)
`model/topology.py`:
```python
@dataclass(frozen=True)
class Topology:
    allowed: np.ndarray            # (N, N) bool, diagonal False
    absorbing: tuple[int, ...]     # rows with no allowed target
def from_config(cfg, n_states) -> Topology   # "full" | "left_to_right" | "cyclic" | {"banded": k} | {"matrix": [[0/1]]}
```
- Shorthands: `full` = all off-diagonal; `left_to_right` = `j > i` (last state absorbing); `cyclic` = `j = (i+1) mod N`; `banded: k` = `0 < |j − i| ≤ k`; `matrix` as given (diagonal must be 0, else `ConfigError`).
- **Masked multinomial logit** (extends E3-T05): for row `r` with current regime `i`, allowed targets `A_i = {j : Ω_ij}`; reference = smallest `j ∈ A_i`; free logits only for `A_i \ {ref}`; `π_rj = 0` (`−inf` in log space) for `j ∉ A_i`. Rows with `|A_i| = 1` have no parameters; rows with `|A_i| = 0` are absorbing. Spec shapes become ragged → implement as one flat vector `transition.a (n_free_logits,)` plus static index arrays mapping into `(R, N)`; covariate coefficients analogous `(n_free_logits, K_pi)`. For order 2, memory tuples are restricted to admissible paths of `Ω` (a tuple `(i, j)` exists only if `Ω_ij`).
- Parameter count = Appendix F: `Σ_i max(0, |A_i| − 1)` per row (× memory tuples for order 2); each transition covariate adds the same count.
- **Absorbing states** (FR-DU02): expanded engine: `a_i(t) ≡ 0` for `i ∈ absorbing` (phase advance disabled; `advance_prob` output overwritten by a static mask in `transition_fn`); duration parameters of absorbing states are dropped from `u` (spec mode forced `fixed`, documented). Explicit engine: `p_i(d) = δ_{d, D_max}` and the segment is right-censored (it never ends before `T`); `log_pmf` row replaced by a static vector.
- Initial distribution: `uniform` over phase-1 states of all regimes (unchanged); topology does not restrict the initial regime (document; constrain via `initial_state: free` + labels if needed).
- Dirichlet priors on rows apply over allowed targets only.
- Gate `model.topology` removed.

## Creates
- `src/hsmmkit/model/topology.py`, edits in `model/transition.py`, `statespace/builders.py`, `statespace/transitions.py`, `engine/explicit_duration.py`
- `tests/unit/test_e12_t03_topology.py`, `tests/property/test_e12_t03_structural_zeros.py`, `tests/golden/test_e12_t03_absorbing.py`

## Acceptance criteria
1. A fitted left-to-right model never decodes a backward jump (Viterbi on synthetic and random data); `A_t` has exact zeros at disallowed entries, rows sum to 1 ≤ 1e-12.
2. Absorbing state: `a_i = 0` in the expanded engine; explicit-engine loglik equals a reference that treats the last segment as censored; both engines agree on a geometric/NB case per Appendix D.
3. `ParamManager.n_free` for transitions equals the Appendix F formula for all shorthands and random matrices (`N ∈ [2,5]`, order 1 and 2).
4. `topology: full` reproduces Part I bitwise (golden).
5. Invalid matrices (non-zero diagonal, wrong shape) → `ConfigError` with path.

## Tests
Unit, property, golden.

## Out of scope
Duration-dependent jumps (E12-T05).

## Connections
- Depends on: E10-T10
- Blocks: E12-T05
- Relates to: E3-T05, E4-T02a/b, E16-T05 (left-to-right preset)
