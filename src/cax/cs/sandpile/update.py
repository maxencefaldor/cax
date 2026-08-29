"""Sandpile update module.

This module implements the toppling rule for the Abelian Sandpile model. When a cell's
chip count reaches or exceeds the threshold (2 * num_spatial_dims), it topples by
distributing one chip to each face-adjacent neighbor.
"""

from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.update import Update


class SandpileUpdate(Update[Array, Array, Array]):
	"""Sandpile toppling rule.

	Applies a single parallel toppling step: all cells at or above the critical
	threshold simultaneously redistribute chips to their face-adjacent neighbors.
	Each critical cell loses ``threshold`` chips, and each face-adjacent neighbor
	of a critical cell gains 1 chip.
	"""

	def __init__(self, *, num_spatial_dims: int = 2, threshold: int | None = None):
		"""Initialize Sandpile update.

		Args:
			num_spatial_dims: Number of spatial dimensions. Used for default threshold.
			threshold: Critical chip count for toppling. Defaults to
				2 * num_spatial_dims (4 in 2D, 6 in 3D).

		"""
		self.threshold = threshold if threshold is not None else 2 * num_spatial_dims

	@override
	def __call__(self, state: Array, perception: Array, input: Array | None = None) -> Array:
		"""Process the current state, perception, and input to produce a new state.

		Performs one parallel toppling step. Cells at or above the threshold topple,
		losing ``threshold`` chips. Each face-adjacent neighbor of a critical cell
		gains 1 chip. If input is provided, it is added to the chip counts before
		evaluating the toppling rule (used for dropping sand grains).

		Args:
			state: Current state (unused, values come from perception).
			perception: Array with shape (..., *spatial_dims, (2*ndim + 1)) containing
				the Von Neumann neighborhood. Channel 0 is the center cell, remaining
				channels are face-adjacent neighbors.
			input: Optional array with shape (..., *spatial_dims, 1) representing
				sand grains to add before toppling.

		Returns:
			Next state with shape (..., *spatial_dims, 1) containing updated chip counts.

		"""
		chips = perception[..., 0:1]
		if input is not None:
			chips = chips + input
		neighbor_chips = perception[..., 1:]

		self_critical = (chips >= self.threshold).astype(jnp.float32)
		neighbor_critical = (neighbor_chips >= self.threshold).astype(jnp.float32)
		num_critical_neighbors = jnp.sum(neighbor_critical, axis=-1, keepdims=True)

		state = chips - self_critical * self.threshold + num_critical_neighbors
		return state
