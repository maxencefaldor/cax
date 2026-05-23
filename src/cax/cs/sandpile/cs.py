"""Sandpile module.

This module implements the Abelian Sandpile model, a cellular automaton exhibiting
self-organized criticality. Chips are stacked on cells of a grid; when any cell
accumulates chips equal to or exceeding a critical threshold, it topples by
distributing one chip to each face-adjacent neighbor. Parallel toppling proceeds
until a stable configuration is reached.
"""

from typing import Literal

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.utils import clip_and_uint8

from .perceive import SandpilePerceive
from .update import SandpileUpdate


class Sandpile(ComplexSystem[Array, Array]):
	"""Abelian Sandpile model.

	A discrete cellular automaton demonstrating self-organized criticality. The state
	is a grid of non-negative integers representing chip counts. When a cell reaches
	the critical threshold, it topples, distributing chips to neighbors. Cascading
	avalanches of topplings produce power-law distributed events.

	Two boundary modes are supported:
		- "CIRCULAR": periodic (toroidal) boundaries conserving total mass.
		- "OPEN": dissipative boundaries where sand falling off the edge is lost,
			which is required for proper self-organized criticality.
	"""

	def __init__(
		self,
		*,
		num_spatial_dims: int = 2,
		threshold: int | None = None,
		padding: Literal["CIRCULAR", "OPEN"] = "CIRCULAR",
	):
		"""Initialize Sandpile.

		Args:
			num_spatial_dims: Number of spatial dimensions (default 2).
			threshold: Critical chip count for toppling. Defaults to
				2 * num_spatial_dims (4 in 2D, 6 in 3D).
			padding: Boundary condition mode. "CIRCULAR" for periodic boundaries,
				"OPEN" for dissipative boundaries (required for SOC).

		"""
		self.num_spatial_dims = num_spatial_dims
		self.threshold = threshold if threshold is not None else 2 * num_spatial_dims
		self.perceive = SandpilePerceive(
			num_spatial_dims=num_spatial_dims,
			padding=padding,
		)
		self.update = SandpileUpdate(
			num_spatial_dims=num_spatial_dims,
			threshold=self.threshold,
		)

	def _step(self, state: Array, input: Array | None = None, *, sow: bool = False) -> Array:
		if input is not None:
			state = state + input

		perception = self.perceive(state)
		next_state = self.update(state, perception)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@nnx.jit
	def render(self, state: Array) -> Array:
		"""Render state to RGB image.

		Maps chip counts to distinct colors using the classic sandpile palette:
		0 chips → dark blue, 1 chip → cyan, 2 chips → yellow, 3 chips → orange.
		Cells at or above the critical threshold are rendered in red.

		Args:
			state: Array with shape (..., *spatial_dims, 1) containing integer chip
				counts stored as float32.

		Returns:
			RGB image with dtype uint8 and shape (..., *spatial_dims, 3).

		"""
		chips = state[..., 0]

		r = jnp.where(
			chips == 0,
			0.1,
			jnp.where(chips == 1, 0.0, jnp.where(chips == 2, 0.9, jnp.where(chips == 3, 1.0, 0.8))),
		)
		g = jnp.where(
			chips == 0,
			0.1,
			jnp.where(chips == 1, 0.7, jnp.where(chips == 2, 0.9, jnp.where(chips == 3, 0.5, 0.0))),
		)
		b = jnp.where(
			chips == 0,
			0.4,
			jnp.where(chips == 1, 0.8, jnp.where(chips == 2, 0.1, jnp.where(chips == 3, 0.0, 0.0))),
		)

		rgb = jnp.stack([r, g, b], axis=-1)
		return clip_and_uint8(rgb)
