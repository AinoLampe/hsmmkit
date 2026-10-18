# Examples and tutorial

`examples/` holds two ready-to-use configurations and a tutorial notebook.

| File | What it is |
|---|---|
| `examples/configs/dax_3state_gbm_nb.yaml` | The reference configuration: 3 regimes, GBM emissions, negative-binomial durations, VIX and the 10y-2y spread as covariates, on `^GDAXI` |
| `examples/configs/spx_3state_gbm_nb.yaml` | The same model on `^GSPC` |
| `examples/tutorial.ipynb` | The Python API end to end: validate, fetch, fit, decode, forecast, backtest, with plots |
| `examples/synthetic.py` | A synthetic look-alike of the index and covariates, used by the notebook and by CI |

## Real data

```bash
regimekit validate-config examples/configs/dax_3state_gbm_nb.yaml
regimekit fetch    examples/configs/dax_3state_gbm_nb.yaml    # needs FRED_API_KEY for VIX / T10Y2Y
regimekit fit      examples/configs/dax_3state_gbm_nb.yaml
regimekit --offline decode   examples/configs/dax_3state_gbm_nb.yaml --model runs/dax_3state_gbm_nb/dax_3state_gbm_nb.rkmodel
regimekit --offline forecast examples/configs/dax_3state_gbm_nb.yaml --model runs/dax_3state_gbm_nb/dax_3state_gbm_nb.rkmodel
regimekit --offline backtest examples/configs/dax_3state_gbm_nb.yaml
```

With the full history since 2000 and 8 restarts a fit takes a while; start with a later `data.primary.start`
and fewer `estimation.restarts` to explore. After the first `fetch` every command can run with `--offline`
against the cached snapshot, which makes a run reproducible.

## Offline and in CI

Provider data is never committed (see the providers guide). CI therefore runs the same six commands on
synthetic data written by `examples/synthetic.py` (`tests/integration/test_e8_t03_e2e.py`), checks that two
fits with the same seed write byte-identical `params.safetensors`, and executes the notebook
(`nox -s examples`). The runs on real `^GDAXI` and `^GSPC` data are `tests/live/test_e8_t03_real_data.py`
(`-m live`, manual, needs network and `FRED_API_KEY`).
