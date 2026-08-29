"""Metrics for Lenia.

Computes spatial statistics of a Lenia state on a toroidal grid using circular statistics.
These metrics drive fitness evaluation, degeneration detection, and state centering in
quality-diversity search loops.
"""

import jax.numpy as jnp
from jax import Array


def metrics_fn(state: Array, *, R: int) -> dict[str, Array]:
	"""Compute spatial statistics of a Lenia state on a toroidal grid.

	Uses circular statistics (mean resultant length) to compute center of mass and
	spatial concentration. The concentration measures how localized the mass distribution
	is on the torus (1 = point mass, 0 = uniform), computed as the product of per-axis
	mean resultant lengths.

	Args:
		state: Lenia state array with shape `(*spatial_dims, channel_size)`.
		R: Kernel radius in grid units.

	Returns:
		Dictionary containing mass, center of mass, and concentration.

	"""
	spatial_dims = state.shape[:-1]
	num_spatial_dims = len(spatial_dims)

	# Per-cell mass (sum over channels) and total mass in grid units
	mass_grid = jnp.sum(state, axis=-1)
	total_mass_grid = jnp.sum(mass_grid)
	mass = total_mass_grid / (R**num_spatial_dims)

	# Circular statistics per spatial dimension
	center_of_mass_grid_list: list[Array] = []
	concentration_list: list[Array] = []
	for i, spatial_dim in enumerate(spatial_dims):
		# Project mass distribution onto axis i
		axes_to_sum = tuple(j for j in range(num_spatial_dims) if j != i)
		mass_i = jnp.sum(mass_grid, axis=axes_to_sum)

		# Circular resultant: treats the axis as a periodic domain [0, 2*pi)
		angles = 2 * jnp.pi * jnp.arange(spatial_dim) / spatial_dim
		resultant = jnp.sum(mass_i * jnp.exp(1j * angles))

		# Center of mass: argument of resultant mapped back to grid coordinates
		center_i = (jnp.angle(resultant) % (2 * jnp.pi)) / (2 * jnp.pi) * spatial_dim
		center_of_mass_grid_list.append(center_i)

		# Concentration: mean resultant length (1 = point mass, 0 = uniform)
		concentration_list.append(jnp.abs(resultant) / total_mass_grid)

	center_of_mass_grid = jnp.array(center_of_mass_grid_list)
	concentration = jnp.prod(jnp.array(concentration_list))

	# Center of mass in physical units
	center_of_mass = center_of_mass_grid / R

	return {
		"mass": mass,
		"mass_grid": mass_grid,
		"center_of_mass": center_of_mass,
		"center_of_mass_grid": center_of_mass_grid,
		"concentration": concentration,
	}


def center_state(state: Array, *, R: int) -> Array:
	"""Center a Lenia state on its center of mass assuming toroidal topology.

	Uses the circular mean to find the center of mass in each spatial dimension,
	then rolls the state so the center of mass is at the grid midpoint.

	Args:
		state: Lenia state array with shape `(*spatial_dims, channel_size)`.
		R: Kernel radius in grid units.

	Returns:
		The state array rolled so that the center of mass is at the center of the grid.

	"""
	metrics = metrics_fn(state, R=R)
	center_of_mass_grid = metrics["center_of_mass_grid"]
	spatial_dims = state.shape[:-1]
	shifts = jnp.stack(
		[
			(spatial_dims[i] // 2 - center_of_mass_grid[i]).astype(int)
			for i in range(len(spatial_dims))
		]
	)
	return jnp.roll(state, shifts, axis=tuple(range(len(spatial_dims))))
