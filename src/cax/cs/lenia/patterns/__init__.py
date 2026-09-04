"""Lenia pattern library.

Ships a small collection of known Lenia creatures as `.npz` archives — plain arrays,
loadable without executing anything, unlike pickle — together with the rule parameters
that sustain them. A pattern is the creature's state stamp; placing it into a larger
grid is the caller's business (see the Lenia example notebook).
"""

from importlib import resources

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..growth import LeniaGrowthParams
from ..kernel import LeniaKernelParams
from ..rule import LeniaRuleParams

PATTERN_NAMES = (
    "5N7KKM",
    "5N7KKM_gyrating",
    "5N7KKM_twin",
    "Gyrorbium",
    "Orbium",
    "VT049W",
)

__all__ = [
    "PATTERN_NAMES",
    "load_pattern",
]


def load_pattern(name: str) -> tuple[Array, LeniaRuleParams]:
    """Load a shipped pattern and the rule parameters that sustain it.

    Args:
        name: Pattern name, one of `PATTERN_NAMES`.

    Returns:
        A `(pattern, rule_params)` tuple: `pattern` has shape
            `(*pattern_spatial_dims, channel_size)` with values in `[0, 1]`, and
            `rule_params` is the `LeniaRuleParams` the creature was catalogued with.

    """
    if name not in PATTERN_NAMES:
        raise ValueError(
            f"Unknown pattern {name!r}. Shipped patterns: {', '.join(PATTERN_NAMES)}"
        )

    path = resources.files("cax.cs.lenia.patterns").joinpath(f"{name}.npz")
    with path.open("rb") as file:
        archive = np.load(file)
        pattern = jnp.asarray(archive["pattern"])
        rule_params = LeniaRuleParams(
            channel_source=jnp.asarray(archive["channel_source"]),
            channel_target=jnp.asarray(archive["channel_target"]),
            weight=jnp.asarray(archive["weight"]),
            kernel_params=LeniaKernelParams(
                r=jnp.asarray(archive["kernel_r"]),
                beta=jnp.asarray(archive["kernel_beta"]),
            ),
            growth_params=LeniaGrowthParams(
                mean=jnp.asarray(archive["growth_mean"]),
                std=jnp.asarray(archive["growth_std"]),
            ),
        )
    return pattern, rule_params
