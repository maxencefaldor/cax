"""Particle Life development module.

Development instantiates a CAX `ParticleLife` from the genotype's attraction matrix and
force shape, simulates it from the genotype's initial arrangement, and observes the
result as centered rendered frames plus toroidal time series.

**The observation problem.** A Lenia creature is a mass field, so "is it alive and
localized" reads straight off mass and concentration. Particle Life conserves its
particles: nothing dies, and a rule that does nothing at all looks exactly as "massive"
as a soliton. What has to be measured instead is *organization* — and a soup and a
soliton differ on four independent axes, each emitted as its own series:

- `concentration` — how localized the arrangement is (1 = a point, 0 = spread over the
  torus). Defined exactly as in Lenia, from the same circular statistics, so it means
  the same thing across systems.
- `clustering` — local density against the uniform expectation: the mean number of
  neighbours within the interaction radius divided by the count a uniform scatter would
  give. 1 *is* the soup; a soliton is many times denser. This is what separates one
  tight cluster from several distant ones, which `concentration` alone cannot see.
- `radius` — the toroidal RMS distance from the center of mass, in interaction radii:
  the size of the structure.
- `polarization` — alignment of the particles' velocities (1 = marching together,
  0 = milling at random). A travelling soliton polarizes; a soup does not.

`thermal_velocity` completes the picture by splitting motion into bulk and random parts:
a soup is all random motion with no net drift, a travelling soliton is the reverse.
"""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from cax.cs.particle_life import ParticleLife, ParticleLifeState

from ...core.motion import MOTION_SERIES, motion_series
from ...core.phenotype import Phenotype
from .config import ParticleLifeConfig
from .genotype import Genotype
from .sample import class_id

# The time series `develop` emits (declared for up-front config validation)
SERIES = (
    "concentration",
    "clustering",
    "radius",
    "polarization",
    "thermal_velocity",
    "center_of_mass",
    *MOTION_SERIES,
)


def develop(
    genotype: Genotype, config: ParticleLifeConfig, *, center: bool = True
) -> Phenotype:
    """Develop a genotype into its phenotype.

    Args:
        genotype: Genotype to develop.
        config: Particle Life configuration.
        center: Whether frames are centered on the arrangement. Descriptors want the
            centered view (translation invariance); visualizations want the raw view,
            which conveys motion.

    Returns:
        The phenotype.

    """
    cs = ParticleLife(
        num_classes=config.num_classes,
        dt=config.dt,
        force_factor=config.force_factor,
        velocity_half_life=config.velocity_half_life,
        r_max=config.r_max,
        beta=genotype.beta,
        attraction_matrix=genotype.attraction,
    )

    state_init = ParticleLifeState(
        class_id=class_id(config),
        position=genotype.position_init,
        velocity=jnp.zeros_like(genotype.position_init),
    )
    _, states = cs(state_init, num_steps=config.num_steps, return_states=True)

    # Sequential over time, not vectorized: the pairwise distances behind `clustering`
    # are (num_particles, num_particles) per step, so vectorizing the whole trajectory
    # would materialize num_steps of them at once
    metrics = jax.lax.map(partial(_metrics, config=config), states)
    # Positions are measured in interaction radii, the system's natural length unit —
    # the analogue of Lenia's kernel radius, so velocities are comparable across systems
    world_size = jnp.full((2,), 1.0 / config.r_max)
    metrics |= motion_series(
        metrics["center_of_mass"], world_size=world_size, T=1.0 / config.dt
    )

    # Render only the observed tail: rendering one frame costs about as much as one
    # simulation step, so drawing a thousand-step run in full would double the budget
    states = jax.tree.map(lambda leaf: leaf[-config.num_frames :], states)
    if center:
        states = jax.vmap(_center_state)(states)
    # Sequential for the same reason: rasterizing one frame holds a
    # (resolution^2, num_particles) distance array
    render = partial(
        cs.render, resolution=config.resolution, particle_radius=config.particle_radius
    )
    frames = jax.lax.map(render, states)

    return Phenotype(frames=frames, series=metrics)


def valid(phenotype: Phenotype, config: ParticleLifeConfig) -> Array:
    """Return False if the phenotype degenerated (dispersed into a soup).

    Particles never die, so there is no analogue of Lenia's mass gate: what is rejected
    here is the loss of *organization*, at every step, on the two axes that a soup and a
    soliton differ on independently — localization and local density.
    """
    concentrated = jnp.all(
        phenotype.series["concentration"] >= config.min_concentration
    )
    clustered = jnp.all(phenotype.series["clustering"] >= config.min_clustering)
    return concentrated & clustered


def _metrics(state: ParticleLifeState, config: ParticleLifeConfig) -> dict[str, Array]:
    """Measure the organization of one arrangement of particles."""
    position, velocity = state.position, state.velocity

    center_of_mass, resultant = _circular_mean(position)

    # Toroidal spread around the center of mass, in interaction radii
    offset = _toroidal_offset(position, center_of_mass)
    radius = jnp.sqrt(jnp.mean(jnp.sum(jnp.square(offset), axis=-1))) / config.r_max

    # Local density against the uniform expectation on the unit torus: pi r^2 (N - 1)
    # neighbours. 1 is exactly the soup; a soliton is many times denser
    separation = _toroidal_offset(position[:, None, :], position[None, :, :])
    distance = jnp.sqrt(jnp.sum(jnp.square(separation), axis=-1))
    num_particles = position.shape[0]
    neighbors = jnp.sum(distance < config.r_max) - num_particles  # drop the self-pairs
    expected = jnp.pi * config.r_max**2 * num_particles * (num_particles - 1)
    clustering = neighbors / expected

    # Motion split into its coherent and random parts
    speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    polarization = jnp.linalg.norm(
        jnp.mean(jnp.where(speed > 0, velocity / speed, 0.0), axis=0)
    )
    deviation = velocity - jnp.mean(velocity, axis=0)
    thermal_velocity = (
        jnp.sqrt(jnp.mean(jnp.sum(jnp.square(deviation), axis=-1))) / config.r_max
    )

    return {
        "concentration": jnp.prod(resultant),
        "clustering": clustering,
        "radius": radius,
        "polarization": polarization,
        "thermal_velocity": thermal_velocity,
        "center_of_mass": center_of_mass / config.r_max,
    }


def _circular_mean(position: Array) -> tuple[Array, Array]:
    """Return the toroidal mean position and the resultant length, per axis.

    The resultant length is the concentration of the axis: 1 when every particle sits at
    the same coordinate, 0 when they are spread evenly around the circle.
    """
    angle = 2 * jnp.pi * position
    cos_mean, sin_mean = (
        jnp.mean(jnp.cos(angle), axis=0),
        jnp.mean(jnp.sin(angle), axis=0),
    )
    center_of_mass = (jnp.arctan2(sin_mean, cos_mean) / (2 * jnp.pi)) % 1.0
    return center_of_mass, jnp.hypot(cos_mean, sin_mean)


def _toroidal_offset(position: Array, origin: Array) -> Array:
    """The shortest displacement from `origin` to `position` on the unit torus."""
    offset = position - origin
    return offset - jnp.round(offset)


def _center_state(state: ParticleLifeState) -> ParticleLifeState:
    """Roll the arrangement so its toroidal center of mass sits mid-world."""
    center_of_mass, _ = _circular_mean(state.position)
    return replace(state, position=(state.position - center_of_mass + 0.5) % 1.0)
