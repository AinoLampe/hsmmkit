# hsmmkit

A domain-neutral JAX library for **hidden semi-Markov models**: regime detection with explicit sojourn-time
distributions, structured state spaces, covariate-driven transitions, probabilistic forecasting, walk-forward
evaluation and a model bundle. The core knows no application domain: data enters through sources (`numpy`,
`pandas`, `csv`, `parquet`, `xarray`), time through clocks, and applications extend it through registries.

Read the [Guide](guide/install.md) to get started, the [Math](math/parameters.md) pages for the governing
formulas and the [Architecture decisions](adr/index.md) for why things are built the way they are.
`regimekit` (market-regime detection and forecasting) is an example extension built on top of `hsmmkit`.
