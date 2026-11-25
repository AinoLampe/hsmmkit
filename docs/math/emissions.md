# Emissions

Each `EmissionFamily` gives the per-step, per-regime log-density of the observation `y_t` given the active
regime `i`, in continuous-time parameters converted to the step's `\Delta_t` (ADR-0001):

$$
\log p(y_t \mid z_t = i, \Delta_t) = \ell_i(y_t; \theta_i, \Delta_t)
$$

## Gaussian (`GaussianFamily`, E10-T06)

Per-state Gaussian with a diagonal covariance, for `M \ge 1` series, with optional time scaling and Itô
correction:

$$
y_t \mid z_t = i \sim \mathcal{N}\!\left(m_i(t)\, s(\Delta_t),\; \operatorname{diag}(\sigma_i(t)^2)\, s(\Delta_t)\right),
\qquad s(\Delta) = \Delta \text{ if } \texttt{scale\_by\_dt} \text{ else } 1
$$

$$
m_i(t) = \mu_i + \gamma_i^\top z^e_t - \left(\tfrac{1}{2}\sigma_i(t)^2 \text{ if } \texttt{ito\_correction}\right),
\qquad \log \sigma_i(t) = \log \sigma_i + \delta_i^\top z^e_t
$$

Options (`model.emission`): `scale_by_dt: false`, `ito_correction: false`, `covariance: diag` (`full` follows with
E11-T02). `native_time` is `continuous` with `scale_by_dt`, else `discrete`. With the defaults the family is the
plain per-step Gaussian `N(\mu_i, \sigma_i^2)`.

Parameters (config keys in brackets): `emission.mean` `(N, M)` [`mean`], `emission.scale` `(N, M)`, positive
[`scale`], `emission.gamma` `(N, K_mean)`, `emission.delta` `(N, K_logscale)`. Emission covariates are selected
by `covariates: {mean: [...], log_scale: [...]}`; `gamma`/`delta` exist only when the corresponding list is
non-empty, with a default `NormalPrior(0, coef_prior_sd)` regularizer. A `constraint: increasing` on `scale`
collapses its spec to `(N,)` (single series, or shared with `per_series: false`): a per-series ordering has no
clean joint semantics, so `from_config` raises `ConfigError` for `increasing` with `n_series > 1`.

The log-density keeps the operation order `\sigma_t = \sigma_i e^{\delta^\top z}`, `\text{var} = \sigma_t^2 \Delta`,
`\text{mean} = (\mu_t - \tfrac12 \sigma_t^2)\Delta`, so that `scale_by_dt: true, ito_correction: true` reproduces
the Part I GBM density bit for bit (an application can register that case under its own name; the
`regimekit` example extension registers it as `gbm`).

## Student-t (`StudentTFamily`, E3-T02)

Heavy-tailed location-scale emission with per-state (or tied) degrees of freedom `\nu_i > 2`:

$$
y_t \mid z_t = i \sim m_i(t)\,\Delta_t + s_i(t)\sqrt{\Delta_t}\; t_{\nu_i}, \qquad
m_i(t) = m_i + \gamma_i^\top z^e_t, \quad \log s_i(t) = \log s_i + \delta_i^\top z^e_t
$$

`s_i` is a **scale, not the standard deviation**: `\mathrm{sd} = s_i \sqrt{\nu_i/(\nu_i - 2)}\,\sqrt{\Delta_t}`.
`m_i` is the location of the observation itself, so there is no `-\sigma^2/2` term; `scale_by_dt` (default `false` in the core) switches the `\Delta_t` scaling on; as `\nu \to \infty` the density
tends to the Gaussian with `ito_correction` and `\mu - \sigma^2/2 = m`, `\sigma = s`. Parameters: `emission.{m,s,nu,gamma,delta}`;
`emission.nu` has shape `(N,)` on the `nu_gt_2` domain, and mode `tied` shares one value (tie group `"nu"`),
i.e. one free coordinate.

## Several series (`ProductEmission`, E9-T01)

With `M > 1` series sharing one regime process, the emissions are independent given the regime,

$$
\log f_i(y_t) = \sum_{m=1}^{M} o_{t,m}\,\log f_i^{(m)}(y_{t,m}), \qquad o_{t,m} \in \{0, 1\}
$$

where `o_{t,m} = 0` marks an unobserved print (`PreparedDataset.mask[t, m]` false): that series adds nothing
at step `t` while the other series keep their density (a step where all series are unobserved adds `0`, as
for `M = 1`). `model.emission.per_series: true` (default) gives every series its own `\mu_{i,m}, \sigma_{i,m}`
(`emission.mu`/`emission.sigma` of shape `(N, M)`); `false` shares one `\mu_i, \sigma_i` per regime across the
series (shape `(N,)`, which also allows `constraint: increasing`). `model/build.py` wraps the family in a
`ProductEmission` only when `M > 1`; single-series models keep the bare family and their results are unchanged.
The master grid is the first (primary) series' timestamps: a missing primary price drops the row, a missing
price of another series masks the returns that need it. Correlated emissions are out of scope.

## Later families

!!! note
    OU/jump/GARCH families are out of scope for 0.1.0 (see E9-T03 spike).

## Symbols and code names

| Math | Code |
|---|---|
| `\mu_i, \sigma_i` | `emission.mu`, `emission.sigma` (GBM) |
| `\gamma_i, \delta_i` | `emission.gamma`, `emission.delta` (drift and log-volatility covariate coefficients) |
| `m_i, s_i, \nu_i` | `emission.m`, `emission.s`, `emission.nu` (Student-t) |
| `\Delta_t` | `PreparedDataset.dt` |
| `z^e_t` | `PreparedDataset.cov_emission` |
| families | `GBMFamily`, `StudentTFamily` (`EMISSION_FAMILIES` registry in `model/build.py`) |
| `o_{t,m}` | `PreparedDataset.mask` (`ProductEmission.log_prob_masked`) |

## Gaussian with a full covariance

With `model.emission.covariance: full` the series of one regime are correlated:

$$y_t \mid i \sim \mathcal N\big(m_i(t)\,s(\Delta_t),\; \Sigma_i(t)\,s(\Delta_t)\big),\qquad
\Sigma_i = L_i L_i^\top,\quad \Sigma_i(t) = e^{2\delta_i^\top z_t} L_i L_i^\top,$$

where `L_i` is lower triangular with a positive diagonal (`softplus` of the unconstrained value) and
`s(Δ) = Δ` with `scale_by_dt`, else 1. The parameters are `emission.mean (N, M)` and the packed factors
`emission.chol (N, M(M+1)/2)` (row-major lower triangle; the `cholesky` domain). The Itô correction subtracts
`½ diag(Σ_i)` from the mean. A scalar prior (`lognormal`) on `chol` acts on the diagonal; `lkj` with `eta`
is the LKJ prior on the correlation matrix.

Closed-form M-step (no covariates, no Itô): with `n = Σγ`, `w = Σγ s`, `S1 = Σγ y`, `S2 = Σγ y yᵀ / s`:
`μ = S1 / w`, `Σ = (S2 − μ S1ᵀ − S1 μᵀ + w μ μᵀ) / n` (a relative jitter of `1e-10` is added before the
Cholesky factorization). The conjugate prior is Normal-Wishart `(m0, κ0, W0, ν0)` (`conjugate_prior()`).
A step with any missing series is dropped (there is no per-series masking under a joint density).

## Multivariate Student-t

A `student_t` configuration with `covariance: full`, or with the parameter names `mean`, `scale` (diagonal) or
`chol` (full) and `nu`, selects the joint multivariate Student-t (the Part I names `m`, `s`, `nu` keep the
independent per-series family):

$$\log f = \log\Gamma\!\big(\tfrac{\nu+M}{2}\big) - \log\Gamma\!\big(\tfrac{\nu}{2}\big) - \tfrac M2\log(\nu\pi)
- \log|\tilde L| - \tfrac{\nu+M}{2}\log\!\big(1 + q/\nu\big),\qquad q = \|\tilde L^{-1}(y - \tilde\mu)\|^2 ,$$

with `L̃ = e^{δᵀz} √s L` and `ν_i > 2` shared by the series of one regime (`2 + softplus`). There is no Itô
correction; `m_step` and `conjugate_prior` are absent (ECM estimates `ν`, E13-T02b). Sampling:
`μ + L̃ ε / √(g/ν)`, `ε ~ N(0, I)`, `g ~ χ²_ν`.
