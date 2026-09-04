"""Phenotype module.

In classical evolutionary terms, a genotype develops into a phenotype, and selection
acts on the phenotype. Here, the phenotype is the observable outcome of simulating a
genotype: rendered frames and named time series. It is the only interface between a
complex system module and the evaluation stack — fitness and descriptor modules consume
phenotypes, never simulation internals — which is what keeps the search core independent
of the complex system. During evaluation, the experiment's encoders enrich `series` with
learned feature series (e.g. per-frame VAE latents) computed from the frames.
"""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Phenotype:
    """Observable outcome of simulating one genotype.

    Attributes:
        frames: RGB frames with dtype uint8 and shape `(num_steps, *spatial_dims, 3)`,
            one per simulation step, centered on the pattern.
        series: Named time series, each with a leading time dimension. Series may have
            different lengths (e.g. velocities are differences of positions) and may be
            scalar per step (mass) or vector per step (center of mass, encoder
            features).

    """

    frames: Array
    series: dict[str, Array]
