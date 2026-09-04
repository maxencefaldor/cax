"""Complex system interface.

A complex system, as the search sees it, is its genotype space and development: four
functions plus the physical metadata the rest of the pipeline needs. Each system module
(`breeder/lenia`, later Flow Lenia, Life, ...) binds its functions to its config in the
config's `build` and hands the search this one value — the core never imports a system.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jax import Array

from .phenotype import Phenotype

# Genotypes are opaque pytrees whose structure only the complex system knows
Genotype = Any


@dataclass(frozen=True)
class ComplexSystem:
    """A complex system bound to its config.

    Attributes:
        sample_fn: `(key) -> Genotype`, a fresh genotype from the prior.
        mutate_fn: `(key, genotype) -> Genotype`, variation.
        develop_fn: `(genotype, *, center=True) -> Phenotype`, genotype to phenotype.
        valid_fn: `(phenotype) -> bool`, False if the phenotype degenerated.
        spatial_dims: Spatial dimensions of the rendered phenotype frames.
        unit: Physical unit in pixels (Lenia: kernel radius `R * state_scale`); fixed
            reference for physically comparable crops across world configurations.
        series: Names of the time series `develop_fn` emits, for validating fitness and
            descriptor series references up front.
        num_frames: Number of frames `develop_fn` emits, from the end of the simulation.
            Systems whose rendering is as costly as their simulation (Particle Life)
            render only the observed tail; Lenia renders every step.

    """

    sample_fn: Callable[[Array], Genotype]
    mutate_fn: Callable[[Array, Genotype], Genotype]
    develop_fn: Callable[..., Phenotype]
    valid_fn: Callable[[Phenotype], Array]

    spatial_dims: tuple[int, int]
    unit: float
    series: tuple[str, ...]
    num_frames: int
