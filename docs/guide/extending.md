# Extending: registering a component

Every named component is registered in `hsmmkit.plugins.REGISTRY` under a *kind* and a *name*:

| Kind | Components today |
|---|---|
| `emission` | `gbm`, `student_t`, `product` |
| `duration` | `geometric`, `negbin`, `poisson`, `gamma`, `weibull`, `lognormal`, `nonparametric` |
| `metric` | the metrics of `hsmmkit.evaluation.metrics` |
| `init` | `rolling_kmeans` (alias `rollvol_kmeans`), `lattice_jitter`, `random` |
| `covariate_forecaster` | `freeze`, `scenario`, `bootstrap`, `ar` |
| `source`, `clock`, `transform`, `preset` | empty until later tickets fill them |

Register a component with the decorator; the name is the one used in the configuration file:

```python
from hsmmkit.plugins import register


@register("init", "my_start")
def my_start(data, spec, key): ...
```

`provenance` (shown in error messages) defaults to the origin of the calling module (`hsmmkit.core`, or the extension's top-level package). Registering a name twice raises a `ConfigError`
that names both provenances; an unknown name raises a `ConfigError` listing every registered name.

Registration happens when the defining module is imported. Reading the registry never writes, so no locking is
needed. Extensions register components; they do not patch core objects.

## Configuration versions

The core configuration (`hsmmkit.config.schema.ProjectConfig`, schema version 1) has no application vocabulary.
Applications extend the core schema
by subclassing its sections and registering migrations per `SCHEMA_ID` (ADR-0016).
