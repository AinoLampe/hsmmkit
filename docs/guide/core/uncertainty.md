# Uncertainty: parametric bootstrap

Hessian standard errors and the χ² likelihood-ratio test rely on large-sample approximations that are poor for
hidden-semi-Markov models: few regime visits, parameters near a boundary, and a different number of regimes
under the null (Hansen 1992). The parametric bootstrap replaces them by simulation.

## Standard errors: `hsmmkit bootstrap`

```bash
hsmmkit bootstrap CONFIG --model runs/demo.hsmm --R 200 --workers 4
```

```python
from hsmmkit.selection import bootstrap_se

res = bootstrap_se(fit, data, R=200, seed=0)  # refits start at theta_hat (refit_from="mle")
res.table  # parameter, index, estimate, se, lower, upper, hessian_se, ratio
res.n_failed  # refits excluded (errors, non-finite log-likelihood; not converged with require_converged)
```

For `r = 1..R` a dataset with the **same lengths, time steps and covariate paths** as the data is simulated from
`theta_hat` and refitted with the same estimator settings. The SE is the sample SD of the refitted constrained
parameters, the interval the 2.5 / 97.5 percentiles. `ratio = se / hessian_se` compares with the Hessian SE of the
fit when it is available (it is not for a singular Hessian or explicit gamma shapes). Refits that land in a
permuted regime labelling are relabelled towards `theta_hat` (the assignment used for MCMC draws).

* `refit_from="mle"` (default) refits from `theta_hat` only; `"init"` uses the configured init methods, which
  also captures start-dependent local optima.
* Replicate `r` uses `key_for(seed, PURPOSE_BOOT, r)` for the data and `derive_seed(seed, PURPOSE_BOOT, r)` for the
  fit, so the table is deterministic per seed and identical with `workers > 1` (a `spawn` process pool).
* Outputs: `bootstrap.parquet`, `bootstrap.md` and a `## Bootstrap` section of `report.md`.
* Supported: fits by `map`, `em` or `gem` of models without autoregressive emissions (the lags of simulated data
  would have to be rebuilt). Observations are simulated in the prepared (standardized) space.

## Nested models: `bootstrap_lrt`

```python
from hsmmkit.selection import bootstrap_lrt

lrt = bootstrap_lrt(fit_n2, fit_n3, data, R=99, seed=0)
lrt.statistic, lrt.p_value
```

`D = 2 (ℓ₁ − ℓ₀)` is compared with `D_r` of `R` datasets simulated under the null model `fit0` and fitted with both
models; `p = (1 + #{D_r ≥ D}) / (R + 1)`. The null is refitted from its optimum, the alternative from its own
optimum **and** the configured init methods, and `D_r` is clipped at 0, so a poor local optimum of the larger
model does not make the null distribution too small. Use it instead of the χ² approximation whenever the models
differ in the number of regimes, or a tested parameter lies on the boundary of its space.

!!! note "Cost"
    `R` refits of one (or two) models: `R = 99` for `N = 2` vs `3` takes minutes on a laptop CPU per test.
    Use `--workers` for the SE bootstrap; each worker compiles its own JIT caches.

Not covered: the non-parametric (block) bootstrap of the observed data.
