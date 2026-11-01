# ADR-0022: Recurrent emissions (carry) and the annualization exception

## Context

GBM and Student-t emissions are memoryless: `log f_i(y_t)` depends on `y_t`, `dt_t` and the parameters only,
so the engines precompute the `(T, N)` emission table before their scans. Markov-switching GARCH
(Haas, Mittnik and Paolella, 2004) needs recurrent state: each regime `i` runs its own variance recursion
`h_{i,t+1} = omega_i + alpha_i eps_{i,t}^2 + beta_i h_{i,t}` with `eps_{i,t} = y_t - m_i dt_t`. The
`EmissionFamily` protocol already reserves `init_carry` / `step_carry` for this. The E9-T03 spike checks that
the engine can use them.

## Decision

1. **Carry branch, chosen on the family, not the data.** `_regime_log_emis` (shared by every engine path)
   calls `family.log_prob_scan(y, mask, dt, theta, z)` when `family.init_carry() is not None`; otherwise it
   calls `log_prob` exactly as before, so GBM / Student-t results are bitwise unchanged
   (`tests/spikes/test_e9_t03_msgarch.py`, plus the E9-T01 golden values).
2. **The recursion runs in its own `lax.scan` ahead of the forward scan, not inside it.** In the Haas form the
   per-regime variances depend on the observed series only, not on the regime path, so they can be computed
   for all regimes at once and the result is an ordinary `(T, N)` table. Forward, backward, Viterbi and the
   parallel scan then share it unchanged. This is the deviation from the ticket sketch (carry threaded in the
   main scan). A family whose carry depends on the regime path (Gray 1996, Klaassen 2002 collapsing schemes)
   would need the carry inside the forward scan and a path-dependent backward pass; that is not covered here.
3. **Masked steps** feed `eps^2 -> h` (the conditional expectation) into the recursion and contribute no
   density (the engine zeroes the row, as for any family).
4. **Annualization exception.** `MSGARCHFamily` is `native_time="discrete"`: `omega`, `alpha`, `beta` are per
   step (CLAUDE.md section 3.1 allows "per step" parameters, never both), `m` is per clock unit and enters
   as `m * dt_t`. The variance recursion does not scale with `dt_t`, so irregular grids make `omega` an
   average over step lengths; use the `steps` clock or a regular grid.
5. **Labelling.** `omega` carries `constraint="increasing"`, which serves as the identifiability guard of the
   fit (no relabelling by `emission.sigma`).

## Consequences

- Experimental: `model/emission/msgarch.py` is not registered in `EMISSION_FAMILIES`, has no config name,
  supports a single series, and `sample` / `cdf` raise `NotImplementedError`. Forecasting, simulation and PIT
  with a GARCH carry are out of scope (design note: they need the realised path inside the sampler's scan).
- A 1-state GARCH(1,1) reproduces a NumPy reference log-likelihood to 1e-6
  (`tests/reference/ref_garch.py`; the `arch` package is not a dependency).
- The associative (parallel) scan and the explicit-duration engine consume the same table, so they work
  unchanged for this family; this is untested for the spike.
- `log_prob` always starts the recursion at the unconditional variance, so one-step uses (predictive scoring on
  a single step) are wrong for this family; the spike does not guard against them.
- A carry family wrapped in `ProductEmission` is not supported (the product branch ignores the carry).

## Status

accepted
