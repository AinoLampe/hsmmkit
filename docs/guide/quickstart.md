# Quickstart

This page runs the whole flow on a small synthetic data set, so it needs no network and no API keys. The
same commands work on real data with `examples/configs/dax_3state_gbm_nb.yaml` (see
[Examples](examples.md)).

## Python

```python
# run from the repository root (examples/ is not an installed package)
from pathlib import Path
import tempfile

from examples.synthetic import write_offline_project  # synthetic index + covariates, csv files
from regimekit import Project

config = write_offline_project(Path(tempfile.mkdtemp()), Path("examples/configs/dax_3state_gbm_nb.yaml"))
p = Project.from_yaml(config)

data = p.fetch()  # PreparedDataset: log-returns, dt, lagged covariates
fit = p.fit(data)  # multi-start MAP estimate (FitResult)
post = p.decode(fit, data)  # filtered / smoothed probabilities, Viterbi path, AIC/BIC/ICL
fc = p.forecast(fit, data, horizon=10)
bt = p.backtest()  # walk-forward evaluation + strategy
```

Methods return results and write nothing unless `write=True`; see [Python API and CLI](project.md).

## Command line

```bash
regimekit validate-config config.yaml
regimekit fetch    config.yaml                       # prices + covariates -> prepared.parquet, quality.json
regimekit fit      config.yaml                       # -> <name>.rkmodel/, fit_report.json
regimekit decode   config.yaml --model runs/<name>/<name>.rkmodel
regimekit forecast config.yaml --model runs/<name>/<name>.rkmodel --horizon 20
regimekit backtest config.yaml                       # -> backtest_*.parquet, strategy.parquet, report.md
regimekit compare  a.yaml b.yaml                     # same data snapshot, side-by-side metrics
```

Global options come before the command: `--out DIR`, `--offline`, `--seed N`, `--device cpu|gpu|auto`,
`--log-level LEVEL`. Exit codes: 0 ok, 2 configuration, 3 data, 4 numerical, 5 provider, 1 anything else.

## Reproducibility

A run is determined by (config, data snapshot, seed, backend). `fetch` records the snapshot ids in
`snapshots.json`; with `--offline` every later command replays the cached snapshot. Two fits of the same
inputs give bitwise-identical parameters on the same hardware and backend.

## Where to go next

- [Configuration reference](config.md) for every option, [Data providers](providers.md) for sources and
  terms of use.
- [Output files](outputs.md) for the schema of everything written.
- [Troubleshooting](troubleshooting.md) when a command fails.
