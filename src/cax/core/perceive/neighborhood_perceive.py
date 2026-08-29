"""Neighborhood perceive base module.

This module provides the shared pad-and-slice implementation for neighborhood-based
perception. Both Moore and Von Neumann perceive modules inherit from this base class,
differing only in which neighbor shifts they enumerate.
"""

from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jax import Array

from .perceive import Perceive

_PADDING_TO_PAD_MODE = {
	"CIRCULAR": "wrap",
	"ZERO": "constant",
	"REFLECT": "reflect",
	"EDGE": "edge",
}


class NeighborhoodPerceive(Perceive[Array, Array]):
	"""Base class for neighborhood-based perception using pad-and-slice.

	Subclasses implement `_get_shifts` to define which neighbor offsets belong to the
	neighborhood. This class handles padding, slicing, and optional reduction over neighbors.
	"""

	def __init__(
		self,
		*,
		num_spatial_dims: int,
		radius: int,
		padding: str = "CIRCULAR",
		include_center: bool = True,
		reduce_fn: Callable[..., Array] | None = None,
	):
		"""Initialize neighborhood perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			radius: Neighborhood radius.
			padding: Boundary condition mode. One of "CIRCULAR" (periodic/torus),
				"ZERO" (zero-padded), "REFLECT" (mirror), or "EDGE" (clamp to boundary).
			include_center: Whether to include the center cell in the output.
			reduce_fn: Optional reduction function applied over the neighbor axis. If None,
				neighbors are concatenated along the channel axis. If provided, it is called
				as `reduce_fn(stacked_neighbors, axis=0)` and the result is concatenated
				with the center (if `include_center` is True).

		"""
		if padding not in _PADDING_TO_PAD_MODE:
			valid = ", ".join(sorted(_PADDING_TO_PAD_MODE.keys()))
			raise ValueError(f"padding must be one of {valid}, got {padding!r}")

		self.num_spatial_dims = num_spatial_dims
		self.radius = radius
		self.padding = padding
		self.include_center = include_center
		self.reduce_fn = reduce_fn

	def _get_shifts(self) -> list[tuple[int, ...]]:
		"""Return the list of neighbor shifts (excluding center).

		Each shift is a tuple of length `num_spatial_dims` with integer offsets
		in the range `[-radius, radius]`.

		Returns:
			List of shift tuples defining the neighborhood.

		"""
		raise NotImplementedError

	@override
	def __call__(self, state: Array) -> Array:
		"""Apply neighborhood perception to the input state.

		The input is assumed to have shape `(..., *spatial_dims, channel_size)`.

		When `reduce_fn` is None, neighbors are concatenated along the channel axis:
			- With `include_center=True`: shape `(..., *spatial, channel_size * (N + 1))`
			- With `include_center=False`: shape `(..., *spatial, channel_size * N)`
			where N is the number of neighbor shifts.

		When `reduce_fn` is provided, neighbors are stacked and reduced:
			- With `include_center=True`: shape `(..., *spatial, 2 * channel_size)`
			- With `include_center=False`: shape `(..., *spatial, channel_size)`

		Args:
			state: State of the cellular automaton.

		Returns:
			Neighborhood perception.

		"""
		spatial_start = state.ndim - self.num_spatial_dims - 1
		spatial_sizes = state.shape[spatial_start : spatial_start + self.num_spatial_dims]

		pad_mode = _PADDING_TO_PAD_MODE[self.padding]
		pad_widths = (
			[(0, 0)] * spatial_start
			+ [(self.radius, self.radius)] * self.num_spatial_dims
			+ [(0, 0)]
		)
		padded = jnp.pad(state, pad_widths, mode=pad_mode)

		shifts = self._get_shifts()

		neighbors = []
		for shift in shifts:
			slices = [slice(None)] * spatial_start
			for dim_idx in range(self.num_spatial_dims):
				start = self.radius - shift[dim_idx]
				slices.append(slice(start, start + spatial_sizes[dim_idx]))
			slices.append(slice(None))
			neighbors.append(padded[tuple(slices)])

		if self.reduce_fn is not None:
			stacked = jnp.stack(neighbors, axis=0)
			reduced = self.reduce_fn(stacked, axis=0)
			if self.include_center:
				return jnp.concatenate([state, reduced], axis=-1)
			return reduced

		if self.include_center:
			return jnp.concatenate([state, *neighbors], axis=-1)
		return jnp.concatenate(neighbors, axis=-1)
