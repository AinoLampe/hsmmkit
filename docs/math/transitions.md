# Transitions

Between-regime jump probabilities, optionally covariate-dependent via a logit link, conditioned on a memory
layout for higher-order chains.

## Memory layout

Row `r` of the jump matrix is a memory tuple: order 1 has `R = N` rows, one per current regime `(i,)`;
order 2 has `R = N(N-1)` rows `(i, j)` with `i != j` (adjacent entries in a memory tuple always differ, since
consecutive entries are distinct regime transitions). `current_of[r]` is the tuple's last element — the
regime whose exit row `r` models — and `target_mask[r, :]` is `True` wherever the target differs from
`current_of[r]`.

## Multinomial logit with reference target (`JumpTransition`)

For row `r` with current regime `i`, targets `j != i` are ordered ascending; the reference target
`ref(i)` is the smallest `j != i` (logit and coefficient implicitly zero):

$$
\pi_{rj}(t) = \frac{\exp(a_{rj} + b_{rj}^\top z^\pi_t)}{\sum_{l \neq i} \exp(a_{rl} + b_{rl}^\top z^\pi_t)},
\qquad a_{r,\mathrm{ref}} = 0,\ b_{r,\mathrm{ref}} = 0
$$

With no transition covariates, the row is parametrized directly as a `simplex_ref`-domain spec
(`transition.pi_rows (R, N-1)`, `N-2` free entries per row via the reference-dropped bijector) rather than
raw logits — this is the same construction, since a constant-logit softmax is exactly a fixed simplex point,
and it gives `ParamManager` the correct free-parameter count directly (Appendix B: `N(N-2)` at order 1,
`N(N-1)(N-2)` at order 2; `N = 2` has zero free parameters — each row is deterministic, matching a 1-element
simplex). With covariates, `transition.a (R, N-2)` / `transition.b (R, N-2, K_pi)` hold the non-reference
logits directly, `NormalPrior(0, coef_prior_sd)`-regularized.

## Generator mode (HMM-only CTMC alternative)

For the geometric-duration, order-1 special case, `transition_matrix_generator` exposes the exact
continuous-time alternative to the discrete-step approximation:

$$
Q_{ij} = \lambda_i \pi_{ij}\ (i \neq j), \qquad Q_{ii} = -\lambda_i, \qquad P_t = \exp(Q \, \Delta_t)
$$

via `jax.scipy.linalg.expm`. The first-order approximation `\lambda_i \Delta_t` (what the discrete-step
jump chain uses) converges to the exact `\exp(Q\Delta_t)` off-diagonal as `\Delta_t \to 0`, with per-entry
error `O((\lambda_{\max}\Delta_t)^2)`.

## Symbols and code names

| Math | Code |
|---|---|
| `\pi_{ij}(t)` | `TransitionModel.jump_probs(theta, z_pi, memory)` |
| `a_{ij}, b_{ij}` | `transition.a`, `transition.b` (with covariates), `transition.pi_rows` (without) |
| `z^\pi_t` | `PreparedDataset.cov_transition` |
| memory rows | `MemoryLayout` in `model/transition.py` |
| generator mode | `model.transition.mode: generator` |

## Topology `Ω` and absorbing regimes

`model.topology` restricts the jump matrix to an allowed set `A_i = {j : Ω_ij}` (diagonal always excluded):
`full` (default, every off-diagonal jump, the Part I parametrization bitwise), `left_to_right` (`j > i`, the last
regime is absorbing), `cyclic` (`j = (i + 1) mod N`), `{banded: k}` (`0 < |j − i| ≤ k`) or `{matrix: [[0/1]]}`
(zero diagonal, `N × N`). For a non-full topology the multinomial logit is masked: the reference of a row is its
smallest allowed target (logit 0), only `A_i \ {ref}` carries logits, and `π_ij = 0` exactly outside `A_i`. The free
logits are one flat vector `transition.a (F,)` (and `transition.b (F, K_π)` with covariates), with
`F = Σ_rows max(0, |A| − 1)` (Appendix F; rows with one allowed target have no parameter). For `order: 2` a memory
tuple `(i, j)` exists only if `Ω_ij` holds. A Dirichlet prior acts over the allowed targets only. The initial
distribution is not restricted by the topology.

A regime with no allowed target is **absorbing**. Expanded engine: its advance probability is forced to 0
(`a_i(t) ≡ 0`: it never advances a phase and never exits); explicit engine: its sojourn pmf is `δ_{d, D_max}` and the
final segment is right-censored (choose `d_max` at least the data length). The duration parameters of absorbing
regimes are `fixed` (they are not part of `u`). A non-full topology identifies the order of the regimes, so
`fit` does not relabel the states afterwards; generator mode (`transition.mode: generator`) needs the full topology.
