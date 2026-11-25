# Model bundle

A fitted model is stored as a directory `<name>.hsmm/` by `hsmmkit.persistence.store.ModelStore`
(ADR-0011). `ModelStore.save(fit, path, cfg, data, extensions=...)` writes it (`ModelStore(suffix=".hsmm")` names the directories); `ModelStore.load(path)` returns the
`FitResult` without needing data. `decode`, `forecast` and `simulate --params model` take such a directory as
`--model DIR`.

| File | Content |
|---|---|
| `manifest.json` | `bundle_version` (2), `package`, `package_version`, `extensions` (`{name: version}`), `jax_version`, `backend`, `devices`, `created_at`, `seed` and `schema_version` of the embedded configuration |
| `config.resolved.yaml` | the full resolved configuration (every default filled) |
| `spec.json` | the model structure (`n_states`, `n_series`, `order`, `d_max`, `engine`, `initial`, family names and classes, `r`), the build inputs taken from the dataset (`n_series`, covariate names per entry point, `clock`, `nominal_dt`) and the expected parameter names and shapes |
| `params.safetensors` | the constrained `theta` (key = parameter name) and the unconstrained `u` (key `__u__`), float64 |
| `prep_stats.json` | the standardization statistics and covariate transforms/lags |
| `provenance.json` | the data snapshot ids |
| `fit_report.json` | `objective`, `loglik`, `converged`, `n_iter`, the restarts table (`init_method`), diagnostics and standard errors |
| `extensions/<name>.json` | opaque payloads of extension packages; the finance application writes its data-source metadata to `extensions/hsmmkit.json` |

`load` rebuilds the `ModelSpec` from the configuration and the stored build inputs through the family
registries, then checks that the stored parameters have the expected names and shapes and equal
`unflatten(u)` to 1e-12. It raises `PersistenceError` for a bundle with a newer `bundle_version`, an
extension package listed in the manifest that cannot be imported (the message names it), a missing file, or a parameter whose name or shape disagrees with the model (the
message names the parameter). On the same backend a reloaded model reproduces `engine.loglik` bit for bit.
The log-likelihood is not recomputed by `load`; the caller checks it against a dataset.

## Bundle version 1

Bundles of `bundle_version 1` (0.1.x and 0.2.x: `snapshots.json`, `profile` in the manifest, `init_strategy` in
the restarts table) still load: `snapshots.json` is read as the provenance and as the payload of the
`hsmmkit` extension (`ModelStore.load_extensions`), `profile` is ignored and the restarts column is renamed
to `init_method`. Saving always writes version 2.
