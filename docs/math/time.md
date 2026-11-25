# Time conventions

Observation times are `\tau_0 < \tau_1 < \dots < \tau_T`; the observation `y_t` (`t = 1..T`) is the log-return
over `(\tau_{t-1}, \tau_t]`. Code index `t` (0-based) is math step `t+1`: row `t` of a `PreparedDataset`
holds `y_{t+1}` together with the covariates that were known at the end of the previous step.

## Clocks and the step size

Every emission and duration component receives the step size `\Delta_t` explicitly. Its unit is set by
`data.clock` (ADR-0013):

| Clock | `\Delta_t` | Parameters are |
|---|---|---|
| `trading` | `1 / (252 H)` years, `H` = `data.clock_options.bars_per_day` | annualized |
| `calendar` | `(\tau_t - \tau_{t-1}) / 365.25` years | annualized |
| `steps` | `1` | per step, never annualized |

Parameters are annualized or per step, never both: under `steps` a "year" is one row, and rates such as
`\lambda` are exit rates per step. Branching on the clock name happens only in `hsmmkit.utils.clock` and
`hsmmkit.data.calendar`; all other library code uses the constants of `hsmmkit.utils.clock`.

## Where the grid must be regular

Everything inside the model accepts an irregular grid (`\Delta_t` varies with the gap) except the
explicit-duration engine, which discretizes sojourn distributions on the nominal step
`\Delta = \operatorname{median}(\Delta_t)` and warns when `\max \Delta_t / \min \Delta_t > 1.5`
(`hsmmkit.utils.clock.is_irregular`, ADR-0007).

## Lags

A covariate used at step `t` must be known at the end of step `t-1`. Preparation enforces `lag >= 1` per
covariate: row `s` carries the last print stamped at or before `\tau_{s-\text{lag}}`, so the covariate
vector `z_t` already is the lagged value. Standardization statistics come from the training window only
and are stored with the model.

## Forecast grid

Forecast steps `T+1..T+h` fall on the next weekdays after `\tau_T` (no holiday calendar in 0.1.0), with
`\Delta_{T+k}` computed per clock by `future_grid`. Under `steps` the timestamps continue with the median
spacing and `\Delta \equiv 1`.

## Symbols and code names

| Math | Code |
|---|---|
| `\Delta_t` | `PreparedDataset.dt` (clock units) |
| nominal `\Delta` | `PreparedDataset.nominal_dt` |
| `y_t` | `PreparedDataset.observations` `(T, M)`, with `mask` |
| `z^\pi_t, z^d_t, z^e_t` | `cov_transition`, `cov_duration`, `cov_emission` `(T, K)` |
| lag | `data.covariates[].lag` |
| `H` | `data.clock_options.bars_per_day` |

## Clocks are registered components (E10-T03)

A clock defines `Δ_t` and the future grid of a forecast. The core registers `steps` (`Δ ≡ 1`) and `calendar`
(`Δ_t` in years from the timestamps); the finance application registers `trading` (`Δ = 1/(252 H)`, `H` =
`data.clock_options.bars_per_day`). Choose the clock with `data.clock`; its options go in `data.clock_options`.
See ADR-0015.
