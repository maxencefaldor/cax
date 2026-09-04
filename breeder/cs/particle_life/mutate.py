"""Particle Life mutation module."""

import jax
from jax import Array

from ...core.variation import mutate_bounded
from .config import ParticleLifeConfig
from .genotype import Genotype


def mutate(key: Array, genotype: Genotype, config: ParticleLifeConfig) -> Genotype:
    """Mutate a genotype with per-parameter constraints."""
    key_attraction, key_beta, key_position = jax.random.split(key, 3)
    mutation_std = config.mutate.mutation_std

    # Attraction and beta: bounded Gaussian, reflected at their ranges
    attraction = mutate_bounded(
        key_attraction,
        genotype.attraction,
        lower=-1.0,
        upper=1.0,
        mutation_std=mutation_std,
    )
    beta = mutate_bounded(
        key_beta,
        genotype.beta,
        lower=config.beta_range[0],
        upper=config.beta_range[1],
        mutation_std=mutation_std,
    )

    # Positions live on the torus, so they *wrap* rather than reflect: a reflection at
    # 0 and 1 would invent a boundary the dynamics do not have
    noise = config.mutate.position_std * jax.random.normal(
        key_position, genotype.position_init.shape
    )
    position_init = (genotype.position_init + noise) % 1.0

    return Genotype(attraction=attraction, beta=beta, position_init=position_init)
