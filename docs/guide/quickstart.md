# Quickstart

Install (`uv sync` in a checkout, or `pip install hsmmkit`) and write a configuration that points at a CSV file:

```yaml
schema_version: 1
project: {name: demo, seed: 7, output_dir: runs/demo}
data:
  clock: steps
  source: {kind: csv, path: data.csv, observation_columns: [x], time_column: t}
model:
  n_states: 2
  emission:
    family: gaussian
    params:
      mean: {mode: free, init: [0.0, 0.0]}
      scale: {mode: free, init: [0.01, 0.03], constraint: increasing}
  duration:
    family: geometric
    params: {rate: {mode: free, init_mean_sojourn_steps: [30, 30]}}
  transition: {mode: jump, order: 1, covariates: []}
estimation: {restarts: 4, init: [rolling_kmeans, lattice_jitter]}
forecast: {horizon: 5, n_sim: 1000, covariate_forecaster: {type: freeze}, quantiles: [0.05, 0.5, 0.95]}
evaluation:
  metrics: [log_score, bic]
  backtest: {scheme: expanding, initial_train_steps: 200, refit_every_steps: 50, horizons: [1]}
output: {write: [backtest]}
```

```bash
hsmmkit validate-config config.yaml
hsmmkit fit      config.yaml                       # -> runs/demo/demo.hsmm/, fit_report.json
hsmmkit decode   config.yaml --model runs/demo/demo.hsmm
hsmmkit forecast config.yaml --model runs/demo/demo.hsmm --horizon 5
hsmmkit backtest config.yaml                       # -> backtest_*.parquet, report.md
```

From Python:

```python
from hsmmkit.project import Project

p = Project.from_yaml("config.yaml")
data = p.load_data()
fit = p.fit(data)
posterior = p.decode(fit, data)
```

See [Data sources](data_sources.md), [Preparation](preparation.md) and [Configuration](config.md).
