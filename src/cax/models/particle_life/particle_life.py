"""Particle Life model."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from cax.core.ca import CA, metrics_fn
from cax.utils.render import clip_and_uint8, hsv_to_rgb

from .particle_life_perceive import ParticleLifePerceive
from .particle_life_update import ParticleLifeUpdate
from .state import State


class ParticleLife(CA):
	"""Particle life model."""

	def __init__(
		self,
		num_classes: int,
		rngs: nnx.Rngs,
		*,
		r_max: float = 0.15,
		beta: float = 0.3,
		dt: float = 0.01,
		velocity_half_life: float = 0.01,
		force_factor: float = 1.0,
		boundary: str = "CIRCULAR",
		metrics_fn: Callable = metrics_fn,
	):
		"""Initialize Particle Life."""
		self.num_classes = num_classes

		key = rngs.params()
		A = jax.random.uniform(key, (num_classes, num_classes), minval=-1.0, maxval=1.0)

		perceive = ParticleLifePerceive(
			A=A,
			r_max=r_max,
			beta=beta,
			force_factor=force_factor,
			boundary=boundary,
		)
		update = ParticleLifeUpdate(
			dt=dt,
			velocity_half_life=velocity_half_life,
			boundary=boundary,
		)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@partial(nnx.jit, static_argnames=("resolution", "particle_radius"))
	def render(
		self,
		state: State,
		resolution: int = 512,
		particle_radius: float = 0.005,
	) -> jax.Array:
		"""Render state to RGB.

		Args:
			state: An array of states containing class_, position, and velocity.
			resolution: Resolution of the output image.
			particle_radius: Radius of particles in the [0, 1] coordinate space.

		Returns:
			Rendered states as an RGB image array of shape (resolution, resolution, 3).

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

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
