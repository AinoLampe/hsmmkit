"""Pure NumPy/SciPy reference implementations (E4-T01): ground truth for engine golden tests.

Every module here is plain, slow, loop-based NumPy/SciPy -- no JAX, no `hsmmkit` import (never;
comparing hsmmkit's JAX engines against a reference that itself calls hsmmkit would make the
golden tests circular, CLAUDE.md §6).
"""
