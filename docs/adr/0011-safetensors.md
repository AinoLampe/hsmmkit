# ADR-0011: safetensors for the model bundle (new runtime dependency)

## Context

A fitted model must be persisted (`<name>.rkmodel/`, E8-T01) so that it reloads without data and reproduces its
log-likelihood bitwise on the same backend. The parameters are named float64 arrays (the constrained `theta`
and the unconstrained vector `u`); the bundle also holds JSON and YAML files. Options for the array file:
NumPy `.npz`, pickle, Arrow/parquet, or `safetensors`.

## Decision

Parameter arrays are stored in `params.safetensors` using the `safetensors` package (`safetensors.numpy`),
keyed by parameter name, with the unconstrained vector under `__u__`. `safetensors` becomes a runtime
dependency (`pyproject.toml`, `uv.lock`). Everything else in the bundle is plain JSON or YAML, so a bundle can
be read without importing regimekit.

## Consequences

- No pickle: loading a bundle never executes code, and a corrupted or hostile file can fail but not run.
- The format is dtype- and shape-explicit (a shape mismatch is detected on load and reported by parameter name),
  language-neutral, memory-mappable and has a stable, specified header, unlike `.npz` (a zip of `.npy`).
- It is a small, widely used, dependency-free wheel. `.npz` would add no dependency, but the temporary
  `.npz` helper used by E6-T02 did not offer the safety and portability properties above.
- The unconstrained `u` is stored next to `theta`; `load` verifies `ParamManager.unflatten(u) == theta` to
  1e-12 and that names and shapes match the rebuilt model.
- Bundle compatibility is governed by `manifest.bundle_version` (currently 1); a newer bundle is refused with a
  clear error rather than guessed at.

## Status

accepted
