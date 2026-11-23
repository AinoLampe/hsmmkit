# E10-T02a — Component registries (emission, duration, metric, source, clock, transform, init, covariate forecaster, preset)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G05, D15, D24 | M7a | Part II §11.2 (D15, D24), §15.1 (`plugins.py`), §15.3 |

## Goal
Replace the ad-hoc name lookups of Part I with one typed registry module so that every named component can be contributed by an extension package.

## Context (self-contained)
`src/regimekit/plugins.py` (becomes `hsmmkit.plugins`):
```python
Kind = Literal["emission", "duration", "metric", "source", "clock", "transform", "init",
               "covariate_forecaster", "preset"]

@dataclass(frozen=True)
class Entry:
    kind: Kind; name: str; obj: Any; provenance: str        # e.g. "regimekit.core", "regimekit.finance"

class Registry:
    def register(self, kind: Kind, name: str, obj: Any, *, provenance: str) -> None: ...   # duplicate → ConfigError naming both provenances
    def get(self, kind: Kind, name: str) -> Any: ...       # unknown → ConfigError(path=None, "unknown <kind> '<name>'; registered: a (pkg), b (pkg)")
    def names(self, kind: Kind) -> tuple[str, ...]: ...    # sorted
    def entries(self, kind: Kind) -> tuple[Entry, ...]: ...
    def _discover(self) -> None: ...                       # no-op hook; entry points in E16-T04

REGISTRY = Registry()
def register(kind: Kind, name: str, *, provenance: str | None = None) -> Callable[[T], T]: ...  # decorator; provenance defaults to the caller's top-level module path
```
- In-process registration only (modules register at import). Core components register from their own module; finance components register from `regimekit/finance/__init__.py`, which is imported lazily by `regimekit/__init__.py` **only when** `HSMM_CORE_ONLY` is unset (keeps the `core-only` job meaningful).
- Re-base existing lookups on the registry, without behaviour change (the Part I config→model assembly lives in `model/build.py` with the dicts `EMISSION_FAMILIES` and `DURATION_FAMILIES`, CLAUDE.md §3.3; they become thin views over `REGISTRY`):
  - emission families (`gbm`, `student_t`, `product`) and duration families (`geometric`, `negbin`, `poisson`, `gamma`, `weibull`, `lognormal`, `nonparametric`) — used by config → `ModelSpec` and by bundle loading (E8-T01);
  - metric registry of E7-T01 (`evaluation/metrics/registry.py` becomes a thin facade over `REGISTRY` kind `metric`; strategy metrics registered with provenance `regimekit.finance` once moved in E10-T07a — until then keep current provenance);
  - provider registry of E2-T01 → kind `source` with names `provider:<name>` is **not** done here (E10-T04b);
  - init strategies (`rollvol_kmeans`, `lattice_jitter`, `random`) → kind `init`;
  - covariate forecasters (`freeze`, `scenario`, `bootstrap`, `ar`) → kind `covariate_forecaster`.
- Kinds `clock`, `transform`, `preset` are created empty and filled by E10-T03, E10-T05, E10-T10.
- Thread safety: registration happens at import; `get` is read-only — no locks needed (document).

## Creates
- `src/regimekit/plugins.py`; registry calls in the modules listed above
- `tests/unit/test_e10_t02a_registry.py`
- `docs/guide/extending.md` (stub: "registering a component")

## Acceptance criteria
1. Duplicate registration raises `ConfigError` naming both provenances; unknown name lists all registered names with provenance.
2. All Part I name lookups go through `REGISTRY` (grep test: no remaining module-level dicts mapping family/metric/init/forecaster names).
3. v0.2.0 golden values (E10-T01) still bitwise; the full Part I suite is green.
4. With `HSMM_CORE_ONLY=1`, `REGISTRY.names("emission")` contains no entry with provenance `regimekit.finance`.

## Tests
Registry unit tests (register/get/names/duplicates/provenance), integration test that config → `ModelSpec` resolves every Part I family through the registry, golden test.

## Out of scope
Config extension and schema v2 (E10-T02b); entry-point discovery (E16-T04).

## Connections
- Depends on: E10-T01
- Blocks: E10-T02b
- Relates to: E7-T01, E8-T01, E16-T04
