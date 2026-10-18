# E0-T05 — JAX runtime bootstrap, PRNG utilities, error hierarchy, structured logging

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | NFR-01, NFR-03, NFR-09 | M0 | §5.1 (principle 6), §6.2 Errors, §6.3 items 1 & 6 |

## Goal
Provide the runtime plumbing every later module imports: x64-enabled JAX, device selection, deterministic keys, typed exceptions and structlog.

## Context (self-contained)
- `utils/jax_setup.py`: `setup(device: Literal["cpu","gpu","auto"] = "auto") -> None`. Sets `jax.config.update("jax_enable_x64", True)`, then verifies `jnp.zeros(1).dtype == jnp.float64`; if not (x64 was disabled by an earlier initialization) raise `NumericalError("x64 disabled ...")`. `auto` picks GPU if `jax.devices("gpu")` is non-empty, else CPU; `gpu` without a GPU raises `NumericalError` with an actionable message. `regimekit/__init__.py` calls `setup()` at import.
- `utils/rng.py`: `key_for(seed: int, *ids: int) -> jax.Array` = `jax.random.key(seed)` followed by `fold_in` for each id in order. `split_n(key, n)`. No global RNG anywhere.
- `utils/errors.py`: `RegimekitError(Exception)`; subclasses `ConfigError, DataError, ProviderError, NumericalError, PersistenceError`; each accepts `path: str | None` (config path or array name) and prefixes it to the message. `EXIT_CODES = {ConfigError: 2, DataError: 3, NumericalError: 4, ProviderError: 5}`.
- `utils/logging.py`: `configure_logging(level: str)`; JSON renderer when env `CI` is set, console renderer otherwise; `get_logger(name)`.
- `utils/typing.py`: re-exports `Array`, `Float`, `Int`, `Bool` from jaxtyping and a `typechecked` decorator that is `jaxtyped(typechecker=beartype)` when `REGIMEKIT_TYPECHECK=1` and identity otherwise.

## Creates
- `src/regimekit/utils/{jax_setup,rng,errors,logging,typing}.py`; `__init__.py` import hook
- `tests/unit/test_e0_t05_*.py`

## Acceptance criteria
1. Importing `regimekit` after `jax.config.update("jax_enable_x64", False)` in a fresh subprocess raises `NumericalError`.
2. `key_for(42, 1, 2)` is bitwise identical across two processes.
3. `NumericalError("bad", path="model.sigma")` message starts with `model.sigma:`; `EXIT_CODES` complete.
4. With `CI=1` log lines are valid JSON; without, human-readable.
5. `typechecked` raises on a wrong-shape call only when `REGIMEKIT_TYPECHECK=1`.

## Tests
Subprocess test for (1); determinism test for (2); log capture for (4); shape check toggle for (5).

## Out of scope
Device selection from config (E1-T02a wires the field; E8-T02 wires the CLI flag).

## Connections
- Depends on: E0-T01
- Blocks: E1-T01, E1-T03a, E1-T04
- Relates to: E0-T02 (`REGIMEKIT_TYPECHECK` in tests)
