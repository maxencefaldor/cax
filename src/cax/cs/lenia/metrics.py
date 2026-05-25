"""Metrics function for Lenia.

This file contains an example implementation of a metrics function specifically designed for Lenia.

The function is responsible for calculating various statistical measures within the Lenia world,
including the mass, center of mass, and other pertinent metrics. These computations are essential
for analyzing the state and dynamics of the Lenia system.
"""

from typing import Any

import jax.numpy as jnp
from jax import Array


def metrics_fn(
	state: Array,
	*,
	R: int,
	active_threshold: float = 0.1,
	empty_fraction: float = 0.01,
	spread_threshold: float = 0.5,
) -> Any:
	"""Metrics function for Lenia.

	Computes spatial statistics of a Lenia state on a toroidal grid, including mass,
	center of mass (via circular mean), spatial concentration, and degeneration flags.

	The spatial concentration is derived from the mean resultant length in circular
	statistics: for each spatial dimension, the magnitude of the complex sum
	``|sum(mass_i * exp(i * 2*pi*x / L))| / sum(mass_i)`` measures how concentrated
	the mass distribution is along that axis (1 = perfectly localized, 0 = uniform).
	The overall concentration is the product across all spatial dimensions.

	Args:
		state: Lenia state array with shape `(*spatial_dims, channel_size)`.
		R: Kernel radius in grid units.
		active_threshold: Threshold for considering a cell active.
		empty_fraction: Fraction of active cells below which the world is empty.
		spread_threshold: Concentration below which the pattern is considered spread.

	Returns:
		Dictionary containing mass, center of mass, concentration, and degeneration flags.

	"""
	spatial_dims = state.shape[:-1]
	num_spatial_dims = len(spatial_dims)

	# Compute mass in grid units
	mass_grid = jnp.sum(state, axis=-1)

	# Compute mass
	total_mass_grid = jnp.sum(mass_grid)
	mass = total_mass_grid / (R**num_spatial_dims)

	# Compute center of mass and per-dimension concentration via circular statistics
	center_of_mass_grid_list: list[Any] = []
	concentration_list: list[Any] = []
	for i, spatial_dim in enumerate(spatial_dims):
		axes_to_sum = tuple(j for j in range(num_spatial_dims) if j != i)
		mass_i = jnp.sum(mass_grid, axis=axes_to_sum)
		x = jnp.arange(spatial_dim)
		complex_sum_i = jnp.sum(mass_i * jnp.exp(1j * 2 * jnp.pi * x / spatial_dim))
		angle_center_i = jnp.angle(complex_sum_i)
		center_i = ((angle_center_i + 2 * jnp.pi) % (2 * jnp.pi)) / (2 * jnp.pi) * spatial_dim
		center_of_mass_grid_list.append(center_i)
		concentration_list.append(jnp.abs(complex_sum_i) / total_mass_grid)
	center_of_mass_grid = jnp.array(center_of_mass_grid_list)
	concentration = jnp.prod(jnp.array(concentration_list))

	# Compute center of mass in physical units
	center_of_mass = center_of_mass_grid / R

	# Check degeneration flags
	active_fraction = jnp.mean(mass_grid > active_threshold)
	is_empty = active_fraction < empty_fraction
	is_spread = concentration < spread_threshold

	return {
		"mass": mass,
		"mass_grid": mass_grid,
		"center_of_mass": center_of_mass,
		"center_of_mass_lattice": center_of_mass_grid,
		"active_fraction": active_fraction,
		"concentration": concentration,
		"is_empty": is_empty,
		"is_spread": is_spread,
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
	center_of_mass_lattice = metrics["center_of_mass_lattice"]
	spatial_dims = state.shape[:-1]
	shifts = tuple(
		(spatial_dims[i] // 2 - center_of_mass_lattice[i]).astype(int)
		for i in range(len(spatial_dims))
	)
	return jnp.roll(state, shifts, axis=tuple(range(len(spatial_dims))))
