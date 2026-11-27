"""ArviZ output of the Bayesian estimators (E13-T06a, ADR-0024).

`to_inference_data` turns draws of constrained parameters into an ArviZ object: a `DataTree` with ArviZ 1.x,
an `InferenceData` with ArviZ 0.x. Groups: `posterior` (one variable per `ParamSpec` name, dims
`(chain, draw, <axis names>)`), `sample_stats` (sampler diagnostics) and `log_likelihood` (per sequence).
ArviZ is imported lazily through `optional_import(..., extra="bayes")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hsmmkit.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["axis_names", "summary_table", "to_inference_data"]

_AXES = ("regime", "series")


def axis_names(name: str, shape: tuple[int, ...]) -> list[str]:
    """Dimension names of the parameter `name`: `regime`, `series`, then `<name>_axis<i>`."""
    dotted = name.replace(".", "_")
    return [_AXES[i] if i < len(_AXES) else f"{dotted}_axis{i}" for i in range(len(shape))]


def _dims_and_coords(
    posterior: Mapping[str, np.ndarray], coords_extra: Mapping[str, Any]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    dims: dict[str, list[str]] = {}
    coords: dict[str, Any] = dict(coords_extra)
    for name, draws in posterior.items():
        shape = tuple(draws.shape[2:])
        names = axis_names(name, shape)
        # a dimension name shared by parameters of different lengths gets a parameter-specific name
        for i, (axis, size) in enumerate(zip(names, shape, strict=True)):
            if axis in coords and len(coords[axis]) != size:
                names[i] = f"{name.replace('.', '_')}_{axis}"
            coords.setdefault(names[i], np.arange(size))
        dims[name] = names
    return dims, coords


def to_inference_data(
    posterior: Mapping[str, np.ndarray],
    sample_stats: Mapping[str, np.ndarray] | None = None,
    log_likelihood: np.ndarray | None = None,
    *,
    sequence_ids: list[str] | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:  # noqa: ANN401
    """Draws (`(chain, draw, *shape)` arrays per parameter name) as an ArviZ object.

    Args:
        posterior: constrained parameters by `ParamSpec` name.
        sample_stats: `(chain, draw)` sampler diagnostics (`diverging`, `tree_depth`, `energy`, ...).
        log_likelihood: `(chain, draw, n_sequences)` per-sequence log-likelihoods.
        sequence_ids: coordinate of the `sequence` dimension.
        attrs: free-form attributes (versions, seed, ...).

    Raises:
        ImportError: ArviZ is not installed (`pip install 'hsmmkit[bayes]'`).
    """
    az = optional_import("arviz", extra="bayes")
    post = {k: np.asarray(v) for k, v in posterior.items()}
    dims, coords = _dims_and_coords(post, {})
    groups: dict[str, Any] = {"posterior": post}
    if sample_stats:
        groups["sample_stats"] = {k: np.asarray(v) for k, v in sample_stats.items()}
    if log_likelihood is not None:
        ll = np.asarray(log_likelihood)
        groups["log_likelihood"] = {"loglik": ll}
        dims["loglik"] = ["sequence"]
        coords["sequence"] = np.asarray(sequence_ids if sequence_ids is not None else range(ll.shape[-1]))
    if int(str(az.__version__).split(".")[0]) < 1:  # ArviZ 0.x
        data = az.from_dict(**groups, dims=dims, coords=coords, attrs=attrs or {})
    else:  # ArviZ 1.x (DataTree)
        data = az.from_dict(groups, dims=dims, coords=coords, attrs=attrs or {})
    return data


def summary_table(idata: Any) -> Any:  # noqa: ANN401
    """`az.summary` of the posterior (a pandas DataFrame with `r_hat`, `ess_bulk`, `ess_tail`)."""
    az = optional_import("arviz", extra="bayes")
    return az.summary(idata)
