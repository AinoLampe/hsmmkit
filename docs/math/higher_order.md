# Higher-order regime memory

Order `k` means the jump distribution out of the current regime depends on the previous `k-1` regimes:

$$
P(q_{n+1} = l \mid q_n = j,\, q_{n-1} = i,\, \dots)
$$

Order 1 is the ordinary HSMM, in which `q_{n+1}` depends on `q_n` only.

## Memory-augmented regimes

For `k = 2` the builder indexes a regime by the pair `(i, j)` with `i \ne j` (previous, current). Transitions
go `(i, j) \to (j, l)` with `l \ne j`, with `\pi_{(i,j)\,l}(t)` the multinomial logit of the transitions page.
The sojourn clock belongs to the current regime `j` (`duration_conditioning: current`, `N` duration rows) or
to the ordered pair (`pair`, `N(N-1)` rows). The builder is generic in `k`: the last `k-1` regimes are the
memory and parameter counts grow as `N (N-1)^{k-1} (N-2)` jump coefficients.

The initial memory at `t = 1` is uniform over the admissible tuples (default) or free
(`model.initial_state`).

## Expanded states and parameter counts

The expanded state `s = (\text{memory}, \text{regime}, \text{phase})` has

$$
S = \sum_{\text{rows } r} r_{i(r)}, \qquad \text{order 1: } S = \sum_i r_i, \qquad \text{order 2: } S = (N-1)\sum_i r_i .
$$

Free parameters without covariates (`p_e`, `p_d` per-state emission and duration parameters):

| Component | Order 1 | Order 2 |
|---|---|---|
| Jump matrix | `N(N-2)` | `N(N-1)(N-2)` |
| Durations (`current`) | `N p_d` | `N p_d` |
| Durations (`pair`) | | `N(N-1)p_d` |
| Emissions | `N p_e` | `N p_e` |

Example, `N = 4`, GBM (`p_e = 2`), negative binomial (`p_d = 1`): order 1 has `8 + 4 + 8 = 20` free parameters,
order 2 has `24 + 4 + 8 = 36`. A transition covariate adds `N(N-2)` (order 1) or `N(N-1)(N-2)` (order 2)
coefficients, a duration covariate `N`, an emission covariate `2N`. Order 2 needs several times more data than
order 1 for the same precision (the recovery suite runs it at 2.5 times the default sample, see the
[validation guide](../guide/validation.md)).

## Symbols and code names

| Math | Code |
|---|---|
| `k` | `model.transition.order` |
| memory tuple | `StateIndex.memory_of` `(S, order-1)` |
| row of the jump matrix | `StateIndex.row_of_state` `(S,)`, `MemoryLayout` in `model/transition.py` |
| `S` | `StateSpace.n_expanded` |
| `current` / `pair` | `model.transition.duration_conditioning`, `StateSpace.n_rates` |
