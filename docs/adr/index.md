# Architecture decisions

Each ADR records a decision, its context, and its consequences, per the format in the repository's
`CONTRIBUTING.md`. Numbers 0001–0010 are the Part I baseline; 0012–0013 record the Part
II amendments folded into the baseline (overview §8). 0011 (safetensors bundle format) is written by E8-T01; 0014+ (reserved for
Part II, or raised during implementation) are added by the tickets that need them.

| ADR | Decision |
|---|---|
| [0001](0001-continuous-time-parameters-discrete-inference.md) | Continuous-time parameters, discrete-time inference |
| [0002](0002-license-and-data-redistribution.md) | Apache-2.0 license; provider data never redistributed |
| [0003](0003-jax-equinox-optax-stack.md) | JAX + Equinox + Optax stack; no TFP |
| [0004](0004-generic-engine-structured-state-index.md) | One generic engine over a structured state index |
| [0005](0005-map-lbfgs-primary-estimator.md) | MAP via L-BFGS is the primary estimator |
| [0006](0006-shifted-negbin-phase-expansion.md) | Shifted negative-binomial phase expansion |
| [0007](0007-explicit-duration-regular-grid.md) | Explicit-duration engine requires a regular grid |
| [0008](0008-approx-inference-plugin.md) | Approximate-inference plug-in interface (factorial models) |
| [0009](0009-pydantic-v2-yaml-config.md) | pydantic v2 + YAML configuration |
| [0010](0010-typer-cli-pandas-pyarrow.md) | typer CLI; pandas in the data layer; pyarrow parquet |
| [0011](0011-safetensors.md) | safetensors for the model bundle (new runtime dependency) |
| [0012](0012-core-profile-split.md) | Core/profile split |
| [0013](0013-three-clocks.md) | Three clocks, including `steps` |
