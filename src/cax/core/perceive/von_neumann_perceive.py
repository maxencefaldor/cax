"""Von Neumann perceive module."""

from collections.abc import Callable
from itertools import product

from jax import Array

from .neighborhood_perceive import NeighborhoodPerceive


class VonNeumannPerceive(NeighborhoodPerceive):
	"""Von Neumann perceive class.

	This class implements perception based on the Von Neumann neighborhood.
	The Von Neumann neighborhood includes all cells within Manhattan distance `radius`
	of the central cell.
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
		"""Initialize Von Neumann perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			radius: Manhattan distance defining the Von Neumann neighborhood extent.
			padding: Boundary condition mode. One of "CIRCULAR" (periodic/torus),
				"ZERO" (zero-padded), "REFLECT" (mirror), or "EDGE" (clamp to boundary).
			include_center: Whether to include the center cell in the output.
			reduce_fn: Optional reduction function applied over the neighbor axis. If None,
				neighbors are concatenated along the channel axis. If provided, it is called
				as `reduce_fn(stacked_neighbors, axis=0)` and the result is concatenated
				with the center (if `include_center` is True).

		"""
		super().__init__(
			num_spatial_dims=num_spatial_dims,
			radius=radius,
			padding=padding,
			include_center=include_center,
			reduce_fn=reduce_fn,
		)

	def _get_shifts(self) -> list[tuple[int, ...]]:
		"""Return all shifts in the Von Neumann neighborhood (excluding center).

		Returns:
			List of shift tuples with Manhattan distance <= radius, excluding the origin.

		"""
		return [
			shift
			for shift in product(range(-self.radius, self.radius + 1), repeat=self.num_spatial_dims)
			if 0 < sum(map(abs, shift)) <= self.radius
		]
