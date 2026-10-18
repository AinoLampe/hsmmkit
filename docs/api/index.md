# API reference

The public surface is `regimekit.Project` (see the [Python API guide](../guide/project.md)); the pages below
document the packages it is built from. Layer rule: `model`, `statespace`, `engine`, `estimation` and
`inference` do not import the data layer except `data.contracts`, and no core package imports
`regimekit.finance` (ADR-0012).

| Page | Packages |
|---|---|
| [Project and CLI](project.md) | `regimekit.project`, `regimekit.cli` |
| [Configuration](config.md) | `regimekit.config` |
| [Data](data.md) | `regimekit.data`, `regimekit.finance` |
| [Model](model.md) | `regimekit.model`, `regimekit.statespace` |
| [Engines and estimation](engines.md) | `regimekit.engine`, `regimekit.estimation` |
| [Inference](inference.md) | `regimekit.inference` |
| [Evaluation](evaluation.md) | `regimekit.evaluation` |
| [Persistence and utilities](persistence.md) | `regimekit.persistence`, `regimekit.utils` |
