# Durations

Each `DurationFamily` gives the sojourn-length pmf `p_i(d)` over `d` steps in regime `i` (geometric, shifted
negative-binomial, Poisson, Gamma, Weibull, log-normal, or nonparametric over `D_max`, per family):

$$
P(\text{sojourn} = d \mid z = i) = p_i(d; \theta_i), \qquad \text{mean\_sojourn}(\theta_i) = \sum_{d} d \, p_i(d)
$$

## Phase representation: `geometric` and `negbin` (E3-T03)

The expanded engine represents a regime's sojourn as a chain of `r_i` phases: the chain enters phase 1 on
regime entry, advances one phase per step with probability `a_i(t)`, and exits (the regime ends) leaving
phase `r_i`. On a constant grid, `d - r_i \sim \text{NegBin}(r_i, a_i)`:

$$
a_i(t) = 1 - \exp(-\lambda_i(t) \, \Delta_t), \qquad
\log \lambda_i(t) = \log \lambda_i + c_i^\top z^d_t
$$

$$
E[d] = \frac{r_i}{a_i}, \qquad \text{Var}[d] = \frac{r_i (1 - a_i)}{a_i^2}, \qquad \min(d) = r_i
$$

`GeometricDuration` is the `r_i = 1` special case (plain HMM sojourn); `advance_prob`/`log_pmf` agree with
`NegBinDuration(r=(1, ..., 1))` exactly, since the advance-probability formula does not depend on `r`.

`\lambda_i` (exit/advance rate, `duration.rate (N,)` positive) is in clock units: per year under the
`trading`/`calendar` clocks, per step under `steps` (`\Delta_t \equiv 1`) — no code branches on the clock
name (CLAUDE.md §3.2). `mean_sojourn(theta, nominal_dt)` reports `E[d] \cdot \text{nominal\_dt}` (renamed
from `mean_sojourn_years`, Part II §15.7/CLAUDE.md §3.1), and `init_from_mean_sojourn(mean_steps,
nominal_dt)` inverts it to a rate: `\lambda_i = r_i / (m_i \cdot \text{nominal\_dt})` (`r_i = 1` for
geometric). `init_mean_sojourn_days` needs a clock/`bars_per_day`-aware days-to-steps conversion that only
`model/build.py` has (CLAUDE.md §3.3); `DurationFamily.from_config` accepts `init_mean_sojourn_steps`
directly and rejects `init_mean_sojourn_days` with a `ConfigError` pointing at that boundary.

A duration covariate link (`duration.c (N, K_d)`, default `NormalPrior(0, coef_prior_sd)`) is evaluated
every step for `advance_prob` (used by the engine's per-step scan) and at a single, fixed entry-time value
`z_entry` for `log_pmf` (the explicit engine's constant-per-segment pmf, ADR-0006's "entry-time evaluation").

## Explicit families (`representation = "explicit"`, E3-T04)

Explicit families give the sojourn pmf `p_i(d)`, `d = 1, \dots, D_{\max}`, on a regular grid of step
`\Delta` (`nominal_dt`, clock units). A continuous sojourn law with cdf `F_i` is discretized as

$$
p_i(d) = \frac{F_i(d\Delta) - F_i((d-1)\Delta)}{F_i(D_{\max}\Delta)}
$$

(truncated at `D_max` and renormalized, so every row sums to 1). Parameters are in clock units, like the
phase families (per year under `trading`/`calendar`, per step under `steps`).

| Family | Parameters | Mean sojourn (clock units) | Entry-time link |
|---|---|---|---|
| `poisson` | `duration.kappa (N,)` | `\Delta + \kappa` (`d = 1 + \text{Poisson}(\kappa/\Delta)`) | `\log \kappa` |
| `gamma` | `duration.shape`, `duration.scale (N,)` | `\text{shape} \cdot \text{scale}` | `\log \text{scale}` |
| `weibull` | `duration.shape`, `duration.scale (N,)` | `\text{scale}\,\Gamma(1 + 1/\text{shape})` | `\log \text{scale}` |
| `lognormal` | `duration.mu_log`, `duration.sd_log (N,)` | `\exp(\mu + \sigma^2/2)` | `\mu_{\log}` |
| `nonparametric` | `duration.logits (N, D_max)`, domain `pmf` | `\Delta \sum_d d\,p(d)` | none |

The link is `c_i^\top z_{\text{entry}}` with `duration.c (N, K_d)`: `log_pmf(theta, d_max, nominal_dt, z_entry)`
returns `(N, D_max)` for a `(K_d,)` entry value and `(T, N, D_max)` for `(T, K_d)`. The `pmf` bijector returns
the constrained simplex values, so `duration.logits` already holds probabilities (`log_pmf` is their log).
Grid cells whose probability underflows are floored at `10^{-300}` so `log` and its gradient stay finite.

`init_from_mean_sojourn(mean_steps, nominal_dt)` uses `mean = \text{mean\_steps} \cdot \Delta`: gamma
`shape = 2`, `scale = \text{mean}/2`; weibull `shape = 1.5`, `scale = \text{mean}/\Gamma(1 + 1/1.5)`; lognormal
`\sigma = 0.5`, `\mu = \log \text{mean} - \sigma^2/2`; poisson `\kappa = (\text{mean\_steps} - 1)\Delta`;
nonparametric a geometric pmf with that mean. In config, `init_mean_sojourn_steps` goes on the family's
primary parameter (`scale`, `mu_log`, `kappa`, `logits`).

**Hazard.** `hazard(log_pmf)` returns `h(d) = p(d)/S(d-1)` with `S(d) = 1 - \sum_{u \le d} p(u)` computed from
the pmf as given; an untruncated geometric pmf has constant hazard `a`, and a pmf renormalized over
`1..D_max` has `h(D_{\max}) = 1`.

**`D_max` rule.** `check_d_max` warns when the truncated mass `1 - F(D_{\max}\Delta)` exceeds `10^{-6}` or
`D_{\max} < 2 \times` the mean sojourn in steps. The phase-family API (`n_phases`, `advance_prob`) raises
`NotImplementedError` for explicit families.

## Symbols and code names

| Math | Code |
|---|---|
| `\lambda_i` | `duration.rate` (phase families) |
| `a_i(t)` | `DurationFamily.advance_prob_t` |
| `r_i` | `model.duration.r` (hyperparameter, not in `u`) |
| `c_i` | `duration.c` (duration covariate coefficients) |
| `\kappa, \text{shape}, \text{scale}, \mu_{\log}, \text{sd}_{\log}` | `duration.kappa`, `duration.shape`, `duration.scale`, `duration.mu_log`, `duration.sd_log` |
| nonparametric pmf | `duration.logits` |
| mean sojourn | `DurationFamily.mean_sojourn(theta, nominal_dt)` |
| families | `GeometricDuration`, `NegBinDuration`, `PoissonDuration`, `GammaDuration`, `WeibullDuration`, `LogNormalDuration`, `NonparametricDuration` |
