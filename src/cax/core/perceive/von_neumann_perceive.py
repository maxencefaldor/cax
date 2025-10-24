"""Von Neumann perceive module."""

from itertools import product

import jax.numpy as jnp

from cax.core import State

from .perceive import Perceive, Perception


class VonNeumannPerceive(Perceive):
	"""Von Neumann perceive class.

	This class implements perception based on the Von Neumann neighborhood.
	The Von Neumann neighborhood includes cells within a specified Manhattan distance of the central
	cell.
	"""

	def __init__(self, num_spatial_dims: int, radius: int):
		"""Initialize Von Neumann perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			radius: Radius for Manhattan distance to compute the Von Neumann neighborhood.

		"""
		self.num_spatial_dims = num_spatial_dims
		self.radius = radius

	def __call__(self, state: State) -> Perception:
		"""Apply Von Neumann perception to the state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The Von Neumann neighborhood for each state.

		"""
		# Init neighbors
		neighbors = [state]

		# Get Moore shifts
		moore_shifts = product(range(-self.radius, self.radius + 1), repeat=self.num_spatial_dims)

		# Get Von Neumann shifts by filtering Moore shifts with Manhattan distance <= radius
		von_neumann_shifts = [
			shift for shift in moore_shifts if 0 < sum(map(abs, shift)) <= self.radius
		]

		# Compute the neighbors
		for shift in von_neumann_shifts:
			neighbors.append(
				jnp.roll(state, shift, axis=tuple(range(-self.num_spatial_dims - 1, -1)))
			)
		return jnp.concatenate(neighbors, axis=-1)
