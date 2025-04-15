"""Particle Lenia model.

[1] https://google-research.github.io/self-organising-systems/particle-lenia/
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from cax.core.ca import CA, metrics_fn
from cax.types import State
from cax.utils.render import clip_and_uint8

from ..lenia.growth import exponential_growth_fn
from .kernel import peak_kernel_fn
from .particle_lenia_perceive import ParticleLeniaPerceive
from .particle_lenia_update import ParticleLeniaUpdate
from .rule import RuleParams


class ParticleLenia(CA):
	"""Particle Lenia model."""

	def __init__(
		self,
		num_spatial_dims: int,
		T: int,
		rule_params: RuleParams,
		*,
		kernel_fn: Callable = peak_kernel_fn,
		growth_fn: Callable = exponential_growth_fn,
		metrics_fn: Callable = metrics_fn,
	):
		"""Initialize Particle Lenia."""
		self.num_spatial_dims = num_spatial_dims
		perceive = ParticleLeniaPerceive(
			num_spatial_dims=num_spatial_dims,
			rule_params=rule_params,
			kernel_fn=kernel_fn,
			growth_fn=growth_fn,
		)
		update = ParticleLeniaUpdate(
			T=T,
		)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@partial(nnx.jit, static_argnames=("resolution", "extent", "particle_radius", "type"))
	def render(
		self,
		state: State,
		resolution: int = 512,
		extent: float = 15.0,
		particle_radius: float = 0.3,
		type: str = "UG",  # Options: "particles", "UG", "E"
	) -> jax.Array:
		"""Render state to RGB.

		Args:
			state: Array of shape (num_particles, num_spatial_dims) containing particle positions.
			resolution: Resolution of the output image.
			extent: Extent of the viewing area.
			particle_radius: Radius of particles.
			type: Type of visualization to show:
				"particles": Only show particles (default)
				"ug": Show kernel and growth fields with particles
				"e": Show energy field with particles

		Returns:
			An array of pixels representing the particle positions and fields.

		"""
		assert self.num_spatial_dims == 2, "Particle Lenia only supports 2D visualization."

		# Create a grid of coordinates
		x = jnp.linspace(-extent, extent, resolution)
		y = jnp.linspace(-extent, extent, resolution)
		grid = jnp.stack(jnp.meshgrid(x, y), axis=-1)  # Shape: (resolution, resolution, 2)

		# Reshape grid for computation
		flat_grid = grid.reshape(-1, 2)

		# Vectorize the field computation over all grid points
		flat_E, flat_U, flat_G = jax.vmap(self.perceive.compute_fields, in_axes=(None, 0))(
			state, flat_grid
		)

		# Reshape back to grid
		E_field = flat_E.reshape(resolution, resolution)
		U_field = flat_U.reshape(resolution, resolution)
		G_field = flat_G.reshape(resolution, resolution)

		# Helper functions for colormapping
		def lerp(x, a, b):
			return a * (1.0 - x) + b * x

		def cmap_e(e):
			stacked = jnp.stack([e, -e], -1).clip(0)
			colors = jnp.array([[0.3, 1.0, 1.0], [1.0, 0.3, 1.0]], dtype=jnp.float32)
			return 1.0 - jnp.matmul(stacked, colors)

		def cmap_ug(u, g):
			vis = lerp(u[..., None], jnp.array([0.1, 0.1, 0.3]), jnp.array([0.2, 0.7, 1.0]))
			return lerp(g[..., None], vis, jnp.array([1.17, 0.91, 0.13]))

		# Calculate particle mask
		distance_sq = jnp.sum(jnp.square(grid[:, :, None, :] - state[None, None, :, :]), axis=-1)
		distance_sq_min = jnp.min(distance_sq, axis=-1)
		particle_mask = jnp.clip(1.0 - distance_sq_min / (particle_radius**2), 0.0, 1.0)

		# Normalize fields for visualization
		_ = (E_field - jnp.min(E_field)) / (jnp.max(E_field) - jnp.min(E_field) + 1e-8)  # E_norm
		U_norm = (U_field - jnp.min(U_field)) / (jnp.max(U_field) - jnp.min(U_field) + 1e-8)
		G_norm = (G_field - jnp.min(G_field)) / (jnp.max(G_field) - jnp.min(G_field) + 1e-8)

		# Create visualizations
		vis_e = cmap_e(E_field)
		vis_ug = cmap_ug(U_norm, G_norm)

		# Apply particle mask
		particle_mask = particle_mask[:, :, None]

		# Create base particle visualization (blue particles on white background)
		vis_particle = jnp.ones((resolution, resolution, 3))
		vis_particle = (
			vis_particle * (1.0 - particle_mask) + jnp.array([0.0, 0.0, 1.0]) * particle_mask
		)

		# Choose visualization based on type
		if type == "UG":
			# Blend particles with UG field
			rgb = vis_ug * (1.0 - particle_mask * 0.7) + vis_particle * (particle_mask * 0.7)
		elif type == "E":
			# Blend particles with E field
			rgb = vis_e * (1.0 - particle_mask * 0.7) + vis_particle * (particle_mask * 0.7)
		else:  # "particles" (default)
			# Just show particles
			rgb = vis_particle

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
