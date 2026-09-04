"""Lenia development module.

Development maps a genotype to its phenotype: the genotype's rule parameters instantiate
a CAX `Lenia` complex system directly, the system is simulated from the genotype's
initial state, and the observable outcome — centered rendered frames and toroidal metric
time series — forms the phenotype.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from cax.cs.lenia import Lenia, center_state, metrics_fn

from ...core.motion import MOTION_SERIES, motion_series
from ...core.phenotype import Phenotype
from .config import LeniaConfig
from .genotype import Genotype

# The time series `develop` emits (declared for up-front config validation)
SERIES = ("mass", "concentration", "center_of_mass", *MOTION_SERIES)


def develop(
    genotype: Genotype, config: LeniaConfig, *, center: bool = True
) -> Phenotype:
    """Develop a genotype into its phenotype.

    Args:
        genotype: Genotype to develop.
        config: Lenia configuration.
        center: Whether frames are centered on the pattern. Descriptors want the
        centered
            view (translation invariance); visualizations want the raw view, which
            conveys motion.

    Returns:
        The phenotype.

    """
    cs = Lenia(
        spatial_dims=config.spatial_dims,
        channel_size=config.channel_size,
        R=config.R,
        T=config.T,
        state_scale=config.state_scale,
        rule_params=genotype.rule_params,
    )
    return observe(cs, genotype.state_init, config, center=center)


def observe(
    cs: Lenia, state_init: Array, config: LeniaConfig, *, center: bool = True
) -> Phenotype:
    """Simulate an instantiated Lenia-family system and observe the phenotype.

    Shared by Lenia and Flow Lenia: the two differ only in the update rule, so the
    simulation, metric series and rendering are identical once the system is built.
    """
    _, states = cs(state_init, num_steps=config.num_steps, return_states=True)

    # Physical unit: the effective kernel radius in pixels, matching the official code
    # (`R = pattern R * world_scale`) — metrics are then invariant to state_scale
    R = config.R * config.state_scale

    metrics = jax.vmap(partial(metrics_fn, R=R))(states)
    metrics = {
        "mass": metrics["mass"],
        "concentration": metrics["concentration"],
        "center_of_mass": metrics["center_of_mass"],
    }
    world_size = jnp.array([dim / R for dim in config.spatial_dims])
    metrics |= motion_series(
        metrics["center_of_mass"], world_size=world_size, T=config.T
    )

    # Render every step
    if center:
        states = jax.vmap(partial(center_state, R=R))(states)
    frames = jax.vmap(cs.render)(states)

    return Phenotype(frames=frames, series=metrics)


def valid(phenotype: Phenotype, config: LeniaConfig) -> Array:
    """Return False if the phenotype degenerated (died out or spread over the torus)."""
    alive = jnp.all(phenotype.series["mass"] >= config.min_mass)
    concentrated = jnp.all(
        phenotype.series["concentration"] >= config.min_concentration
    )
    return alive & concentrated
