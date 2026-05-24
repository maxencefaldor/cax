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
	full_fraction: float = 0.5,
) -> Any:
	"""Metrics function for Lenia.

	Args:
		state: Lenia state array with shape `(*spatial_dims, channel_size)`.
		R: Kernel radius in grid units.
		active_threshold: Threshold for considering a cell active.
		empty_fraction: Fraction of active cells below which the world is empty.
		full_fraction: Fraction of active cells above which the world is full.

	Returns:
		Dictionary containing mass, center of mass, and degeneration flags.

	"""
	spatial_dims = state.shape[:-1]
	num_spatial_dims = len(spatial_dims)

	# Compute mass in grid units
	mass_grid = jnp.sum(state, axis=-1)

	# Compute mass
	mass = jnp.sum(mass_grid) / (R**num_spatial_dims)

	# Compute center of mass in grid units using circular mean for each dimension
	center_of_mass_grid_list: list[Any] = []
	for i, spatial_dim in enumerate(spatial_dims):
		axes_to_sum = tuple(j for j in range(num_spatial_dims) if j != i)
		mass_i = jnp.sum(mass_grid, axis=axes_to_sum)
		x = jnp.arange(spatial_dim)
		angle_center_i = jnp.angle(jnp.sum(mass_i * jnp.exp(1j * 2 * jnp.pi * x / spatial_dim)))
		center_i = ((angle_center_i + 2 * jnp.pi) % (2 * jnp.pi)) / (2 * jnp.pi) * spatial_dim
		center_of_mass_grid_list.append(center_i)
	center_of_mass_grid = jnp.array(center_of_mass_grid_list)

	# Compute center of mass in physical units
	center_of_mass = center_of_mass_grid / R

	# Check if world is empty or full
	active_fraction = jnp.mean(mass_grid > active_threshold)
	is_empty = active_fraction < empty_fraction
	is_full = active_fraction > full_fraction

	return {
		"mass": mass,
		"mass_grid": mass_grid,
		"center_of_mass": center_of_mass,
		"center_of_mass_lattice": center_of_mass_grid,
		"active_fraction": active_fraction,
		"is_empty": is_empty,
		"is_full": is_full,
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
