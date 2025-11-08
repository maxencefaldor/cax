"""Particle Life module.

This module implements Particle Life, a particle-based system where particles of different
types (classes) interact through pairwise attraction and repulsion forces. The interaction
strength between particle types is defined by an attraction matrix, creating diverse emergent
behaviors such as clustering, orbiting, and complex self-organizing structures. Unlike Boids,
Particle Life uses distance-dependent forces that can be both attractive and repulsive,
enabling richer dynamics and pattern formation.
"""

from functools import partial

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input
from cax.utils.render import clip_and_uint8, hsv_to_rgb

from .perceive import ParticleLifePerceive
from .state import ParticleLifeState
from .update import ParticleLifeUpdate


class ParticleLife(ComplexSystem):
	"""Particle Life class."""

	def __init__(
		self,
		num_classes: int,
		*,
		dt: float = 0.01,
		force_factor: float = 1.0,
		velocity_half_life: float = 0.01,
		r_max: float = 0.15,
		beta: float = 0.3,
		A: Array,
	):
		"""Initialize Particle Life.

		Args:
			num_classes: Number of distinct particle types (classes). Each type can have
				different interactions with other types as specified in the attraction matrix.
			dt: Time step of the simulation in arbitrary time units. Smaller values
				produce smoother motion but require more steps for the same duration.
			force_factor: Global scaling factor for all interaction forces. Higher values
				create stronger, more dynamic interactions.
			velocity_half_life: Time constant for velocity decay due to friction. After
				this time, velocity is halved without force input. Smaller values create
				more damped, viscous dynamics.
			r_max: Maximum interaction distance in coordinate space [0, 1]. Particles beyond
				this distance do not interact. Larger values increase computation cost.
			beta: Distance threshold parameter controlling the transition from repulsion to
				attraction. Typically in range [0, 1], where smaller values create stronger
				short-range repulsion.
			A: Attraction matrix of shape (num_classes, num_classes) where A[i, j] defines
				the attraction strength from type i to type j. Positive values attract,
				negative values repel. Values typically range from -1 to 1.

		"""
		self.num_classes = num_classes

		self.perceive = ParticleLifePerceive(
			force_factor=force_factor,
			r_max=r_max,
			beta=beta,
			A=A,
		)
		self.update = ParticleLifeUpdate(
			dt=dt,
			velocity_half_life=velocity_half_life,
		)

	def _step(
		self, state: ParticleLifeState, input: Input | None = None, *, sow: bool = False
	) -> ParticleLifeState:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@partial(nnx.jit, static_argnames=("resolution", "particle_radius"))
	def render(
		self,
		state: ParticleLifeState,
		*,
		resolution: int = 512,
		particle_radius: float = 0.005,
	) -> Array:
		"""Render state to RGB image.

		Renders particles as colored circles on a black background. Each particle type
		(class) is assigned a distinct hue from the color spectrum, with colors evenly
		distributed across the HSV color space. Particles are drawn with smooth anti-aliased
		edges based on their distance from pixel centers. The visualization uses 2D coordinates
		in the range [0, 1].

		Args:
			state: ParticleLifeState containing class_, position, and velocity arrays.
				Position should have shape (num_particles, 2) with coordinates in [0, 1].
				Class array determines the color of each particle.
			resolution: Size of the output image in pixels for both width and height.
				Higher values produce smoother, more detailed renderings.
			particle_radius: Radius of each particle in coordinate space [0, 1]. Particles
				appear as smooth circles with this radius. Larger values make particles more
				visible but may cause overlap.

		Returns:
			RGB image with dtype uint8 and shape (resolution, resolution, 3), where
				particles appear as colored circles on a black background, with colors
				determined by particle type.

		"""
		assert state.position.shape[-1] == 2, "Particle Life only supports 2D visualization."

		# Create grid of pixel centers
		x = jnp.linspace(0, 1, resolution)
		y = jnp.linspace(0, 1, resolution)
		grid = jnp.stack(jnp.meshgrid(x, y), axis=-1)  # Shape: (resolution, resolution, 2)

		# Compute squared distances to all particles
		positions = state.position  # Shape: (num_particles, 2)
		distance_sq = jnp.sum(
			(grid[:, :, None, :] - positions[None, None, :, :]) ** 2, axis=-1
		)  # Shape: (resolution, resolution, num_particles)

		# Find minimum squared distance and index of closest particle
		min_distance_sq = jnp.min(distance_sq, axis=-1)  # Shape: (resolution, resolution)
		closest_particle_idx = jnp.argmin(distance_sq, axis=-1)  # Shape: (resolution, resolution)

		# Get class of the closest particle for each pixel
		closest_class = state.class_[closest_particle_idx]  # Shape: (resolution, resolution)

		# Compute smooth mask based on distance to closest particle
		mask = jnp.clip(
			1.0 - min_distance_sq / (particle_radius**2), 0.0, 1.0
		)  # Shape: (resolution, resolution)

		# Generate colors for each class using HSV
		hues = jnp.linspace(0, 1, self.num_classes, endpoint=False)
		hsv = jnp.stack([hues, jnp.ones_like(hues), jnp.ones_like(hues)], axis=-1)
		colors = hsv_to_rgb(hsv)  # Shape: (num_classes, 3)

		# Assign colors based on closest particle's class
		particle_colors = colors[closest_class]  # Shape: (resolution, resolution, 3)

		# Create black background
		background = jnp.zeros((resolution, resolution, 3))  # Shape: (resolution, resolution, 3)

		# Blend particle colors with background using the mask
		rgb = (
			background * (1.0 - mask[..., None]) + particle_colors * mask[..., None]
		)  # Shape: (resolution, resolution, 3)

		return clip_and_uint8(rgb)
