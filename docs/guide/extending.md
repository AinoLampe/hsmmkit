# Extending: registering a component

Every named component is registered in `regimekit.plugins.REGISTRY` under a *kind* and a *name*:

| Kind | Components today |
|---|---|
| `emission` | `gbm`, `student_t`, `product` |
| `duration` | `geometric`, `negbin`, `poisson`, `gamma`, `weibull`, `lognormal`, `nonparametric` |
| `metric` | the metrics of `regimekit.evaluation.metrics` |
| `init` | `rolling_kmeans` (alias `rollvol_kmeans`), `lattice_jitter`, `random` |
| `covariate_forecaster` | `freeze`, `scenario`, `bootstrap`, `ar` |
| `source`, `clock`, `transform`, `preset` | empty until later tickets fill them |

Register a component with the decorator; the name is the one used in the configuration file:

```python
from regimekit.plugins import register


@register("init", "my_start")
def my_start(data, spec, key): ...
```

`provenance` (shown in error messages) defaults to the origin of the calling module (`regimekit.core`,
`regimekit.finance`, or the extension's top-level package). Registering a name twice raises a `ConfigError`
that names both provenances; an unknown name raises a `ConfigError` listing every registered name.

Registration happens when the defining module is imported. Reading the registry never writes, so no locking is
needed. Extensions register components; they do not patch core objects.

## Configuration versions

The core configuration (`regimekit.config.schema.ProjectConfig`) has no finance vocabulary. The finance
application uses `regimekit.finance.FinanceConfig`, schema version 2: the source of the observations is
`data.source` (`kind: provider` plus the provider options), the clock options are `data.clock_options`
(`bars_per_day`), and the preparation keys are `observation_transform` and `missing_observations`. Version-1
files (with `profile`, `data.primary`, `bars_per_day`, `initial_train_years`, ...) are migrated automatically
when loaded; the resolved dump written next to a run is always version 2. Applications extend the core schema
by subclassing its sections and registering migrations per `SCHEMA_ID` (ADR-0016).
