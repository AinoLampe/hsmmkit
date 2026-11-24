"""Environment that makes the v0.2.0 golden pipeline independent of the CPU's vector extensions.

Golden values are compared bitwise, but numpy's SIMD dispatch, glibc's libm variants, OpenBLAS kernels and
XLA's CPU code generation choose their instruction set from the host CPU. The runner pool mixes CPU types
(an AVX-512 machine and an AVX2 machine gave different bits). Every process that computes golden values runs
with the instruction set capped at AVX2 + FMA3, a level every supported machine has (E10-T01, owner
decision (b)). This module does not import numpy: the variables must be in the environment before the
process starts, which `scripts/make_golden_v0_2_0.py` ensures by re-executing itself (`os.execve`).
"""

from __future__ import annotations

import os

PIN_FLAG = "REGIMEKIT_GOLDEN_PINNED"

_AVX512_NUMPY = (
    "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
)
_AVX512_GLIBC = (
    "-AVX512F,-AVX512CD,-AVX512BW,-AVX512DQ,-AVX512VL,-AVX512_VNNI,-AVX512_BF16,-AVX512_FP16,-AVX512IFMA"
)

PINNED_ENV: dict[str, str] = {
    PIN_FLAG: "1",
    "NPY_DISABLE_CPU_FEATURES": _AVX512_NUMPY,
    "GLIBC_TUNABLES": f"glibc.cpu.hwcaps={_AVX512_GLIBC}",
    "OPENBLAS_CORETYPE": "Haswell",
    "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "XLA_FLAGS": "--xla_cpu_max_isa=AVX2",
    "JAX_PLATFORMS": "cpu",
    "REGIMEKIT_TYPECHECK": "1",
}


def pinned_environ(base: dict[str, str] | None = None) -> dict[str, str]:
    """`base` (default `os.environ`) with the pinned variables set."""
    env = dict(os.environ if base is None else base)
    env.update(PINNED_ENV)
    return env


def is_pinned() -> bool:
    """Whether this process was started with the pinned environment."""
    return os.environ.get(PIN_FLAG) == "1"
