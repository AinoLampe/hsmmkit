# ADR-0026: Online parameter updates — recursive EM (Cappé & Moulines) on the expanded engine

Status: accepted (E13-T08). Decision: **go** for E15-T05 with the Cappé–Moulines online EM on the
**expanded engine**, closed-form components only, `kappa = 0.7`.

## Context

`OnlineFilter` (E15-T01) filters with fixed parameters. Streams drift, and a user may want the parameters to follow
the data. The requirement (FR-ON05) is an *optional*, deterministic update that builds on the filter and on the EM
sufficient statistics (E13-T01a / E13-T02a) without a second inference code path.

The recursion (Cappé & Moulines 2009; Cappé 2011) keeps running expected sufficient statistics. With the filter
`alpha_t`, the backward kernel `phi_t(i | j) = alpha_{t-1}(i) A_ij / sum_i' alpha_{t-1}(i') A_i'j`, the step
`g_t = t^-kappa`, `kappa in (0.5, 1]`, and the per-step statistics `s(i, j, y_t)`:

```
rho_t(j) = sum_i phi_t(i | j) [ (1 - g_t) rho_{t-1}(i) + g_t s(i, j, y_t) ]
S_t      = sum_j alpha_t(j) rho_t(j)        theta_t = m_step(S_t)  for t >= t_min
```

`spikes/online_em.py` is a NumPy prototype of exactly this for Gaussian emissions and geometric durations
(`A_ii = 1 - a_i`, `A_ij = a_i pi_ij`; the M-step maps the transition counts back to `(a_i, pi_ij)` as
`estimation.em` does). `tests/spikes/test_e13_t08_online_em.py` (slow marker) carries the evidence below.

## Evidence (prototype, 2 regimes, mean sojourn 25, rough start: means and scales off by 30 %)

| Check | Result |
|---|---|
| Frozen parameters, `g_t = 1/t` (the fastest admissible decay): `t S_t` vs the offline expected statistics | equal to `1e-8` against an independent NumPy forward-backward, and to `3e-12` (relative; the test asserts 1 %) against `expanded_estep` (`gamma_regime`, `stay`): the recursion is *exact* for fixed `theta`, which is the "reproduces offline EM statistics" requirement |
| Tracking a regime-1 `sigma` that drifts +50 % over `T = 20 000` (mean abs. relative error after `t = 2000`, 5 seeds) | `kappa = 0.7`: 2.1 - 3.5 % (mean 2.7 %); `kappa = 0.6`: 3.4 - 4.3 %; one run: `0.55`: 4.4 %, `0.8`: 3.1 %; **`kappa = 1`: 10.6 %** (no forgetting: the average lags the drift) — all but `kappa = 1` below the 10 % bound |
| Stationary data, `kappa = 1`, one pass over `T = 20 000` | `sigma = (0.502, 1.205)` for truth `(0.5, 1.2)`, means `(-0.002, 1.492)` for `(0, 1.5)` |
| Cost of the NumPy prototype | `O(N^2 K)` per step with `K = N^2 + 3 N`; 54 microseconds per step for `N = 2` (pure NumPy, one CPU core) |

## Decisions

1. **Algorithm: Cappé–Moulines online EM.** Alternatives considered: *mini-batch / block EM* (re-run the E-step on a
   window every `m` steps: simple and reuses `expanded_estep`, but it is not a per-step update, doubles the code
   paths and forgets with a hard window); *SGD on the filtered log-likelihood* (works for families without a closed
   M-step but needs forward sensitivities `O(S p)` per step and step-size tuning per parameter scale). Block EM is
   kept as a possible follow-up for non-closed-form components; SGD is not planned.
2. **Supported components (E15-T05).** Expanded engine only. Updated online: every component whose family has
   `sufficient_stats` / `m_step` (the same capability discovery as `EMEstimator`: Gaussian diagonal and full,
   Poisson, Bernoulli, categorical, mixtures and products of them; Student-t has no closed-form M-step and stays
   frozen), geometric and fixed-`r` negative-binomial
   durations (stay / advance counts per rate row), jump matrices and the initial distribution (frozen after the
   first step: it does not matter on a stream). **Frozen at the fitted values:** components without a closed-form
   M-step (`m_step` returns `None`), covariate links (`gamma`, `delta`, `a`, `b`, `c`: no closed form), priors
   (online EM is the maximum-likelihood recursion; `use_priors` is ignored with a warning), and everything on the
   **explicit engine** — its `(regime, elapsed)` state would make `rho` `(N D_max, K)` and the update
   `O((N D_max)^2 K)` per step, and nonparametric duration pmfs have no per-step form. An unsupported model raises a
   `ConfigError` naming the component.
3. **Defaults.** `kappa = 0.7` (inside `(0.5, 1]`, best of the measured values; `kappa = 1` is allowed and means
   "stationary data"), burn-in `t_min = max(500, 25 * n_free)` observations with the parameters held fixed while
   the statistics accumulate, `update_every = 1` (an integer `> 1` runs the M-step every `k` observations and counts
   observations, never calls), floors `sigma >= 1e-3` of the data scale, occupancy floor `1e-8` per state (a regime
   that is never visited keeps its previous parameters). Configuration under `online.em: {kappa, burn_in,
   update_every}`; absent = no online updates (the current behaviour).
4. **`OnlineState` layout (state version 2).** Added to the arrays of `state.safetensors`: `rho (S, K)` float64
   (`K` = size of the statistics vector: `2 R_rate + R N` transition / rate counts plus the family statistics),
   `stats (K,)`, `u (n_free,)` the *current* unconstrained parameter vector, and the integer `n_updates`; the
   metadata gains `online_em: {kappa, burn_in, update_every}`. Size `8 (S K + K + n_free)` bytes, e.g. `S = 6`,
   `K = 40`: about 2 kB. `spec_hash` is unchanged: it ties the state to the model *structure and parameter layout*,
   not to the values, so a state whose parameters moved still loads against its fit; the saved `u` wins over the
   bundle's parameters on load (`load_state` documents it). Version-1 states keep loading (no online EM, parameters
   from the bundle).
5. **Reproducibility.** The update is part of the single jitted `OnlineFilter.update` step (same function for
   `push` and `push_many`), so streaming one row at a time, `push_many` and a save / load round trip in the middle
   are **bitwise equal** to each other; the schedule counts observations (`t`), not calls. Equality with the *offline*
   E-step is numerical (`1e-8`, different summation order), not bitwise: the offline-filter bitwise guarantee of
   E15-T01 holds for fixed parameters only (`online.em` absent).
6. **Drift detection and alerts.** The alert rules of E15-T03 keep reading regime probabilities and are unchanged.
   The parameter trajectory is exposed instead: `OnlineStep` gains `theta_change` (`||u_t - u_{t-1}||_2`, `0`
   before `t_min`) and the stream CLI can emit the current constrained parameters with `--emit-params`. A
   parameter-drift alert (relative change over a window above a tolerance) is a follow-up, not part of E15-T05.

## Complexity

| Quantity | Cost per observation |
|---|---|
| Filter (E15-T01) | `O(S^2)` dense, less with the structured `transition_fn` |
| `rho` update `phi^T rho` | `O(S^2 K)` (dense); the transition / rate columns are one-hot per `(i, j)`, so only the family block needs the full product |
| `S_t` and M-step | `O(S K)` and the family's closed form (`O(N M^2)` for a full covariance) |
| Memory | `O(S K)` |

## Consequences

* E15-T05 implements decision 1-6 with the existing `m_step` / `sufficient_stats` protocol methods and adds no
  dependency; its acceptance tests are the prototype's: tracking error `< 10 %` on the drifting-`sigma` series,
  exactness for frozen parameters against `expanded_estep`, stream equality (push / `push_many` / save / load).
* The explicit engine, covariate links and non-closed-form families stay frozen online; users who need them refit
  (the walk-forward / backtest path) or wait for the block-EM follow-up.
* `kappa` is data dependent (forgetting horizon `~ t^kappa`): the guide must say that a smaller `kappa` follows
  faster drift and is noisier, and that `kappa = 1` converges to the batch fixed point on stationary data.
