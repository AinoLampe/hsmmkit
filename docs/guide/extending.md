# Extending: registering a component

Every named component is registered in `regimekit.plugins.REGISTRY` under a *kind* and a *name*:

| Kind | Components today |
|---|---|
| `emission` | `gbm`, `student_t`, `product` |
| `duration` | `geometric`, `negbin`, `poisson`, `gamma`, `weibull`, `lognormal`, `nonparametric` |
| `metric` | the metrics of `regimekit.evaluation.metrics` |
| `init` | `rollvol_kmeans`, `lattice_jitter`, `random` |
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
