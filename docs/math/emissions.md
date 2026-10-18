# Emissions

Each `EmissionFamily` gives the per-step, per-regime log-density of the observation `y_t` given the active
regime `i`, in continuous-time parameters converted to the step's `\Delta_t` (ADR-0001):

$$
\log p(y_t \mid z_t = i, \Delta_t) = \ell_i(y_t; \theta_i, \Delta_t)
$$

## GBM (`GBMFamily`, E3-T01)

Per-state geometric Brownian motion: Gaussian log-returns scaled by `\Delta_t`, with linear drift and
log-volatility covariate links.

$$
y_t \mid z_t = i \sim \mathcal{N}\!\left(\left(\mu_i(t) - \tfrac{1}{2}\sigma_i(t)^2\right)\Delta_t,\; \sigma_i(t)^2 \Delta_t\right)
$$

$$
\mu_i(t) = \mu_i + \gamma_i^\top z^e_t \qquad
\log \sigma_i(t) = \log \sigma_i + \delta_i^\top z^e_t
$$

`\mu_i`/`\sigma_i` are in clock units: annualized under the `trading`/`calendar` clocks (`\mu` per year,
`\sigma` per `\sqrt{\text{year}}`), per step under `steps` (where `\Delta_t \equiv 1`, so the density
collapses to `N(\mu_i - \sigma_i^2/2, \sigma_i^2)`) — no code branches on the clock name (CLAUDE.md §3.2).

`GBMFamily` declares `support = "real"` and `scale_by_dt = True`. Emission covariates (`z`, the
`cov_emission` block) may drive drift and log-volatility through different column subsets, selected by
`drift_idx`/`logvol_idx`; `gamma`/`delta` are present only when the corresponding covariate list is
non-empty, with a default `NormalPrior(0, coef_prior_sd)` regularizer (`emission.gamma`, `emission.delta`).

`GBMFamily.from_config` maps `model.emission` config to a configured instance plus its `ParamSpec`s
(`emission.mu`, `emission.sigma`, and `emission.gamma`/`emission.delta` when configured). The `increasing`
bijector (`ParamManager`) only supports a 1-D `(N,)` cell, so a `constraint: increasing` on `sigma`
collapses its spec to `(N,)` rather than `(N, M)` — a per-series ordering constraint has no clean joint
semantics. This matches Appendix A's single-series (`M = 1`) usage; `from_config` raises `ConfigError`
for `constraint: increasing` with `n_series > 1` rather than silently sharing one ordered value across
every series.

## Student-t (`StudentTFamily`, E3-T02)

Heavy-tailed location-scale emission with per-state (or tied) degrees of freedom `\nu_i > 2`:

$$
y_t \mid z_t = i \sim m_i(t)\,\Delta_t + s_i(t)\sqrt{\Delta_t}\; t_{\nu_i}, \qquad
m_i(t) = m_i + \gamma_i^\top z^e_t, \quad \log s_i(t) = \log s_i + \delta_i^\top z^e_t
$$

`s_i` is a **scale, not the standard deviation**: `\mathrm{sd} = s_i \sqrt{\nu_i/(\nu_i - 2)}\,\sqrt{\Delta_t}`.
`m_i` is the location of the log-return itself, so there is no `-\sigma^2/2` term; as `\nu \to \infty` the density
tends to GBM with `\mu - \sigma^2/2 = m`, `\sigma = s`. Parameters: `emission.{m,s,nu,gamma,delta}`;
`emission.nu` has shape `(N,)` on the `nu_gt_2` domain, and mode `tied` shares one value (tie group `"nu"`),
i.e. one free coordinate.

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
