"""Sandpile perceive module.

This module implements the perception function for the Abelian Sandpile model.
It gathers the chip counts of face-adjacent neighbors needed for the toppling rule,
supporting both periodic (circular) and open (dissipative) boundary conditions.
"""

from typing import Literal

import jax.numpy as jnp
from jax import Array

from cax.core.perceive import Perceive
from cax.core.perceive.perceive import Perception


class SandpilePerceive(Perceive[Array]):
	"""Sandpile perception.

	Gathers the Von Neumann neighborhood (face-adjacent cells) for each cell.
	In 2D, this produces 5 channels: [self, N, S, E, W] (center + 4 neighbors).

	Supports two boundary modes:
		- "CIRCULAR": periodic boundaries via jnp.roll (sand wraps around edges).
		- "OPEN": dissipative boundaries where neighbors outside the grid are zero
			(sand falls off the edge when boundary cells topple).
	"""

	def __init__(
		self,
		*,
		num_spatial_dims: int = 2,
		padding: Literal["CIRCULAR", "OPEN"] = "CIRCULAR",
	):
		"""Initialize Sandpile perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions (default 2).
			padding: Boundary condition mode. "CIRCULAR" for periodic boundaries,
				"OPEN" for dissipative boundaries where sand leaves the system.

		"""
		self.num_spatial_dims = num_spatial_dims
		self.padding = padding

	def _get_neighbor(self, state: Array, axis: int, direction: int) -> Array:
		"""Get the neighbor along a given axis and direction.

		Args:
			state: State array.
			axis: Spatial axis index (0-indexed from the first spatial dim).
			direction: -1 or +1 indicating the shift direction.

		Returns:
			Shifted state representing the neighbor values.

		"""
		true_axis = axis - self.num_spatial_dims - 1
		rolled = jnp.roll(state, direction, axis=true_axis)

		if self.padding == "OPEN":
			idx = [slice(None)] * state.ndim
			if direction == -1:
				idx[state.ndim + true_axis] = slice(-1, None)
			else:
				idx[state.ndim + true_axis] = slice(0, 1)
			rolled = rolled.at[tuple(idx)].set(0.0)

		return rolled

	def __call__(self, state: Array) -> Perception:
		"""Apply Sandpile perception to the state.

		Args:
			state: State with shape (..., *spatial_dims, 1).

		Returns:
			Perception with shape (..., *spatial_dims, 2*num_spatial_dims + 1) containing
				the center cell and face-adjacent neighbor values.

		"""
		neighbors = [state]
		for axis in range(self.num_spatial_dims):
			neighbors.append(self._get_neighbor(state, axis, -1))
			neighbors.append(self._get_neighbor(state, axis, +1))

		return jnp.concatenate(neighbors, axis=-1)
