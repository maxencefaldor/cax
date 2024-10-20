"""Moore Perceive module."""

from itertools import product

import jax.numpy as jnp

from cax.core.perceive.perceive import Perceive
from cax.types import Perception, State


class MoorePerceive(Perceive):
	"""Moore Perceive class.

	This class implements perception based on the Moore neighborhood.
	The Moore neighborhood includes cells that are within a certain distance from the central cell
	in all dimensions simultaneously.
	"""

	def __init__(self, num_spatial_dims: int, radius: int):
		"""Initialize the Moore Perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			radius: Radius for Manhattan distance to compute the Moore neighborhood.

		"""
		self.num_spatial_dims = num_spatial_dims
		self.radius = radius

	def __call__(self, state: State) -> Perception:
		"""Apply Moore perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The Moore neighborhood for each state, with the central cell first.

		"""
		# Init neighbors
		neighbors = [state]

		# Get Moore shifts
		moore_shifts = [
			shift
			for shift in product(range(-self.radius, self.radius + 1), repeat=self.num_spatial_dims)
			if shift != (0,) * self.num_spatial_dims
		]

		# Compute the neighbors
		for shift in moore_shifts:
			neighbors.append(jnp.roll(state, shift, axis=tuple(range(-self.num_spatial_dims - 1, -1))))

		return jnp.concatenate(neighbors, axis=-1)
