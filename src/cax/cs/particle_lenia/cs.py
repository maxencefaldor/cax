"""Particle Lenia module.

This module implements Particle Lenia, a particle-based variant of Lenia where discrete
particles interact through continuous field potentials. Unlike grid-based Lenia, particles
move freely in continuous space and experience forces derived from kernel and growth fields
computed from neighboring particles.

References:
	[1] Particle Lenia and the energy-based formulation, Mordvintsev et al. 2022.
		https://google-research.github.io/self-organising-systems/particle-lenia/

"""

from collections.abc import Callable
from typing import Literal, override

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.utils import clip_and_uint8, nearest_point, pixel_grid, soft_disk_mask

from .growth import ParticleLeniaGrowthParams, peak_growth_fn
from .kernel import ParticleLeniaKernelParams, peak_kernel_fn
from .perceive import ParticleLeniaPerceive
from .rule import ParticleLeniaRuleParams
from .state import ParticleLeniaState
from .update import ParticleLeniaUpdate


class ParticleLenia(ComplexSystem[ParticleLeniaState, Array]):
	"""Particle Lenia class."""

	def __init__(
		self,
		*,
		num_spatial_dims: int,
		T: float,
		kernel_fn: Callable[[Array, ParticleLeniaKernelParams], Array] = peak_kernel_fn,
		growth_fn: Callable[[Array, ParticleLeniaGrowthParams], Array] = peak_growth_fn,
		rule_params: ParticleLeniaRuleParams,
	):
		"""Initialize Particle Lenia.

		Args:
			num_spatial_dims: Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
				Determines the dimensionality of particle positions and field computations.
			T: Time resolution controlling the temporal discretization. Higher values
				produce smoother temporal dynamics with smaller update steps.
			kernel_fn: Callable that computes pairwise kernel weights between particles
				based on their distance. Takes rule parameters and returns kernel values.
			growth_fn: Callable that maps kernel field values to growth field values.
				Defines how particles respond to their local neighborhood density.
			rule_params: Instance of ParticleLeniaRuleParams containing kernel and growth
				parameters such as radii, peak positions, widths, and heights.

		"""
		self.num_spatial_dims = num_spatial_dims
		self.perceive = ParticleLeniaPerceive(
			num_spatial_dims=num_spatial_dims,
			kernel_fn=kernel_fn,
			growth_fn=growth_fn,
			rule_params=rule_params,
		)
		self.update = ParticleLeniaUpdate(
			T=T,
		)

	@override
	def _step(self, state: ParticleLeniaState, input: Array | None = None) -> ParticleLeniaState:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		return next_state

	@nnx.jit(static_argnames=("resolution", "extent", "particle_radius", "mode"))
	@override
	def render(
		self,
		state: ParticleLeniaState,
		*,
		resolution: int = 512,
		extent: float = 15.0,
		particle_radius: float = 0.3,
		mode: Literal["particles", "UG", "E"] = "UG",
	) -> Array:
		"""Render state to RGB image.

		Renders Particle Lenia state as particles optionally overlaid on field visualizations.
		Particles appear as blue circles. The background can show kernel field (U), growth field
		(G), or energy field (E) to visualize the underlying dynamics driving particle motion.
		Field visualizations use color mapping to represent field intensities across space.

		Args:
			state: ParticleLeniaState containing particle positions in continuous space.
				Currently only 2D visualization is supported.
			resolution: Size of the output image in pixels for both width and height.
				Higher values produce smoother field gradients but increase computation cost.
			extent: Half-width of the viewing area in coordinate space. The view spans from
				-extent to +extent in each dimension. Adjust to zoom in or out on the particle
				system.
			particle_radius: Radius of each particle in coordinate space. Particles are drawn
				as smooth circles with anti-aliased edges.
			mode: Visualization mode determining what fields to display:
				"particles": Only show particles on white background.
				"UG": Show particles overlaid on kernel (U) and growth (G) field
					visualization (default).
				"E": Show particles overlaid on energy field visualization.

		Returns:
			RGB image with dtype uint8 and shape (resolution, resolution, 3), showing particles
				and optionally the underlying field structure that drives their motion.

		"""
		if self.num_spatial_dims != 2:
			raise ValueError("Particle Lenia only supports 2D visualization.")

		# Create a grid of coordinates
		grid = pixel_grid(resolution, low=-extent, high=extent)  # (resolution, resolution, 2)

		# Reshape grid for computation
		flat_grid = grid.reshape(-1, 2)

		# Vectorize the field computation over all grid points. nnx transforms reject
		# bound methods, so the unbound method is vmapped with the module as first arg.
		flat_U, flat_G, flat_R = nnx.vmap(
			ParticleLeniaPerceive.compute_fields, in_axes=(None, None, 0)
		)(self.perceive, state, flat_grid)

		# Reshape back to grid; the energy field is repulsion minus growth
		U_field = flat_U.reshape(resolution, resolution)
		G_field = flat_G.reshape(resolution, resolution)
		R_field = flat_R.reshape(resolution, resolution)
		E_field = R_field - G_field

		# Helper functions for colormapping
		def lerp(x: Array, a: Array, b: Array) -> Array:
			return a * (1.0 - x) + b * x

		def cmap_e(e: Array) -> Array:
			stacked = jnp.stack([e, -e], -1).clip(0)
			colors = jnp.array([[0.3, 1.0, 1.0], [1.0, 0.3, 1.0]], dtype=jnp.float32)
			return 1.0 - jnp.matmul(stacked, colors)

		def cmap_ug(u: Array, g: Array) -> Array:
			vis = lerp(u[..., None], jnp.array([0.1, 0.1, 0.3]), jnp.array([0.2, 0.7, 1.0]))
			return lerp(g[..., None], vis, jnp.array([1.17, 0.91, 0.13]))

		# Calculate particle mask
		distance_sq_min, _ = nearest_point(grid, state.position)
		particle_mask = soft_disk_mask(distance_sq_min, particle_radius)

		# Normalize fields for visualization
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

		# Choose visualization based on mode
		if mode == "UG":
			# Blend particles with UG field
			rgb = vis_ug * (1.0 - particle_mask * 0.7) + vis_particle * (particle_mask * 0.7)
		elif mode == "E":
			# Blend particles with E field
			rgb = vis_e * (1.0 - particle_mask * 0.7) + vis_particle * (particle_mask * 0.7)
		elif mode == "particles":
			rgb = vis_particle
		else:
			raise ValueError(f"mode must be one of 'particles', 'UG', 'E', got {mode!r}")

		return clip_and_uint8(rgb)
