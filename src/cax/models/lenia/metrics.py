"""Metrics function for Lenia.

This file contains an example implementation of a metrics function, `metrics_fn`, specifically
designed for the Lenia model.

The function is responsible for calculating various statistical measures within the Lenia world,
including the mass, center of mass, and other pertinent metrics. These computations are essential
for analyzing the state and dynamics of the Lenia system.
"""

import jax.numpy as jnp

from cax.types import Input, Metrics, Perception, State


def metrics_fn(
	next_state: State,
	state: State,
	perception: Perception,
	input: Input,
	*,
	R: int,
	active_threshold: float = 0.1,
	empty_fraction: float = 0.01,
	full_fraction: float = 0.5,
) -> Metrics:
	"""Metrics function for Lenia."""
	spatial_dims = next_state.shape[:-1]
	num_spatial_dims = len(spatial_dims)

	# Compute mass in grid units
	mass_grid = jnp.sum(next_state, axis=-1)

	# Compute mass
	mass = jnp.sum(mass_grid) / (R**num_spatial_dims)

	# Compute center of mass in grid units using circular mean for each dimension
	center_of_mass_grid = []
	for i, spatial_dim in enumerate(spatial_dims):
		axes_to_sum = tuple(j for j in range(num_spatial_dims) if j != i)
		mass_i = jnp.sum(mass_grid, axis=axes_to_sum)
		x = jnp.arange(spatial_dim)
		angle_center_i = jnp.angle(jnp.sum(mass_i * jnp.exp(1j * 2 * jnp.pi * x / spatial_dim)))
		center_i = ((angle_center_i + 2 * jnp.pi) % (2 * jnp.pi)) / (2 * jnp.pi) * spatial_dim
		center_of_mass_grid.append(center_i)
	center_of_mass_grid = jnp.array(center_of_mass_grid)

	# Computer center of mass in physical units
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
