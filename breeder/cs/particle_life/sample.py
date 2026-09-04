"""Particle Life sampling module.

Genotype sampling strategies. Every strategy has the same signature
`(key, config) -> Genotype` and is selected by name through `config.sample.strategy`:

- `uniform`: particles scattered over the whole torus — the classical Particle Life
  start, from which structure must self-organize out of a soup.
- `blob`: particles inside one disk — the localized start, from which a soliton can
  *persist*. The analogue of seeding Lenia with a catalogued creature: it hands the
  search a candidate soliton and asks it to keep it alive.

Class labels are assigned round-robin rather than at random: the classes are balanced by
construction, so a genotype's behaviour comes from its attraction matrix instead of from
an accidental class imbalance.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .config import ParticleLifeConfig
from .genotype import Genotype


def sample(key: Array, config: ParticleLifeConfig) -> Genotype:
    """Sample a fresh genotype using the configured strategy."""
    return SAMPLE_FNS[config.sample.strategy](key, config)


def uniform(key: Array, config: ParticleLifeConfig) -> Genotype:
    """Random rule and particles scattered over the whole torus."""
    key_rule, key_position = jax.random.split(key)
    position_init = jax.random.uniform(key_position, (config.num_particles, 2))
    return _with_rule(key_rule, config, position_init)


def blob(key: Array, config: ParticleLifeConfig) -> Genotype:
    """Random rule and particles inside one disk at the center of the torus.

    Sampled uniformly *by area* (radius as the square root of a uniform draw), so the
    disk has even density instead of a center-heavy one.
    """
    key_rule, key_radius, key_angle = jax.random.split(key, 3)
    radius = config.sample.blob_radius * jnp.sqrt(
        jax.random.uniform(key_radius, (config.num_particles,))
    )
    angle = jax.random.uniform(key_angle, (config.num_particles,), maxval=2 * jnp.pi)
    offset = jnp.stack([radius * jnp.cos(angle), radius * jnp.sin(angle)], axis=-1)
    return _with_rule(key_rule, config, (0.5 + offset) % 1.0)


def _with_rule(
    key: Array, config: ParticleLifeConfig, position_init: Array
) -> Genotype:
    """Complete a genotype with a rule sampled from the prior."""
    key_attraction, key_beta = jax.random.split(key)
    attraction = jax.random.uniform(
        key_attraction,
        (config.num_classes, config.num_classes),
        minval=-1.0,
        maxval=1.0,
    )
    beta = jax.random.uniform(
        key_beta, minval=config.beta_range[0], maxval=config.beta_range[1]
    )
    return Genotype(attraction=attraction, beta=beta, position_init=position_init)


def class_id(config: ParticleLifeConfig) -> Array:
    """The round-robin class label of every particle: balanced by construction."""
    return jnp.arange(config.num_particles) % config.num_classes


SAMPLE_FNS = {
    "uniform": uniform,
    "blob": blob,
}
