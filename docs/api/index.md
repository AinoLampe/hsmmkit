# API reference

The public surface is `hsmmkit.Project` (see the [Python API guide](../guide/project.md)); the pages below
document the packages it is built from. Layer rule: `model`, `statespace`, `engine`, `estimation` and
`inference` do not import the data layer except `data.contracts`, and the core imports no application package (ADR-0012, ADR-0014).

| Page | Packages |
|---|---|
| [Project and CLI](project.md) | `hsmmkit.project`, `hsmmkit.cli` |
| [Configuration](config.md) | `hsmmkit.config` |
| [Data](data.md) | `hsmmkit.data` |
| [Model](model.md) | `hsmmkit.model`, `hsmmkit.statespace` |
| [Engines and estimation](engines.md) | `hsmmkit.engine`, `hsmmkit.estimation` |
| [Inference](inference.md) | `hsmmkit.inference` |
| [Evaluation](evaluation.md) | `hsmmkit.evaluation` |
| [Persistence and utilities](persistence.md) | `hsmmkit.persistence`, `hsmmkit.utils` |
