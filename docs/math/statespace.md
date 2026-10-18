# State space

`build_expanded(spec) -> StateSpace` (E4-T02a) enumerates the expanded state space `s = (memory, regime,
phase)` a `ModelSpec` describes, and the index arrays the engines (`E4-T03a/b`, `E4-T06a/b`) and simulator
(`E6-T01`) need to work with it without re-deriving this structure.

## Memory tuples and the `S` formula

A memory tuple has length `order` and no two adjacent entries equal (consecutive regimes always differ);
its last element is the *current* regime, whose sojourn phase advances. Reusing `TransitionModel`'s own row
enumeration (`build_memory_layout`, E3-T05) keeps the jump-matrix row indexing and the expanded-state memory
indexing consistent by construction: expanded-state memory is exactly a row's tuple with the current regime
dropped.

$$
S = \sum_{\text{rows } r} r_{i(r)}, \qquad
\text{order 1: } S = \sum_i r_i, \qquad \text{order 2: } S = (N-1) \sum_i r_i
$$

since at order 1 there is one row per regime, and at order 2 every regime `i` is the "current" regime of
exactly `N-1` rows (one per distinct predecessor).

## Index arrays

`StateIndex` gives `regime_of`, `phase_of` (1-based), `memory_of (S, order-1)` and the one-hot collapse
matrix `C (S, N)` (`C[s, regime_of[s]] = 1`), plus `row_of_state (S,)` — the jump-matrix row governing exits
from `s`. `StateSpace` adds:

- **`entry_state_of (R, N)`** — the `phase == 1` expanded state entered when a jump from row `r`'s regime to
  `target` occurs: `entry_state_of[r, j]` is the state `(shift(m_r, i_r), j, 1)`, where
  `shift(m, i) = (m[1:], i)` slides the memory window forward to include the regime just left
  (`shift` is the empty tuple at order 1, since there is no memory to slide); `-1` where `j` equals `i_r`
  (that's the *advance* case within the same regime, not a jump).
- **`rate_index_of (S,)` / `n_rates`** — which row of the duration parameter array (`n_rates` rows) governs
  state `s`'s sojourn: `regime_of[s]` (`n_rates = N`) under `duration_conditioning: current`, or the ordered
  pair `(memory_of[s][-1], regime_of[s])` (`n_rates = N(N-1)`, requires `order >= 2`) under `pair`.
- **`phase1_idx (S_phase1,)`** — the ascending indices of every `phase == 1` state, used to scatter the
  initial distribution onto the full `(S,)` support.

## Initial distribution

`initial_log_rho(spec, space, theta, nominal_dt) -> (S,)` is `-inf` everywhere except `phase == 1` states:
`uniform` puts equal mass `1 / S_phase1` there; `free` scatters the `initial.logits` spec's constrained
`simplex_ref` values (already probabilities, not raw logits, by the time `ParamManager` hands them to
`theta`) onto the same support. `stationary` (E4-T02b) is the stationary distribution of
`A_bar = transition_fn(theta, z=0, nominal_dt)`, spanning every state (not just `phase == 1` — the chain
starts "already running"), found by power iteration on `A_bar` (`pi_{k+1} = normalize(pi_k A_bar)`) rather
than `jnp.linalg.eig`: a general matrix's eigenvector is only defined up to an arbitrary sign, so naively
clipping it non-negative zeroes the whole vector (`0/0 = NaN` on normalizing) whenever the sign comes back
negative, and `jax.grad` through non-symmetric eigenvectors is unsupported by JAX. Power iteration has
neither problem and is fully differentiable, at the cost of a fixed iteration count (500) rather than an
exact closed form; `π A_bar = π` still holds within tolerance for the well-conditioned, aperiodic chains
this domain produces.

## Symbols and code names

| Math | Code |
|---|---|
| `N` | `model.n_states` |
| `S` | `StateSpace.n_expanded` |
| `D_{max}` | `model.d_max` |
| regime / phase of state `s` | `StateIndex.regime_of`, `StateIndex.phase_of` (1-based) |
| `A_t` | built per step by `statespace.transitions` (never materialized as `(T, S, S)`) |
