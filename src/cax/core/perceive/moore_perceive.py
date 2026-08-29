"""Moore perceive module."""

from collections.abc import Callable
from itertools import product
from typing import override

from jax import Array

from .neighborhood_perceive import NeighborhoodPerceive


class MoorePerceive(NeighborhoodPerceive):
	"""Moore perceive class.

	This class implements perception based on the Moore neighborhood.
	The Moore neighborhood includes all cells within Chebyshev distance `radius` of the
	central cell — i.e., the full hypercube of side length `2 * radius + 1` excluding
	the center.
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
		"""Initialize Moore perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			radius: Chebyshev distance defining the Moore neighborhood extent.
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

	@override
	def _get_shifts(self) -> list[tuple[int, ...]]:
		"""Return all shifts in the Moore neighborhood (excluding center).

		Returns:
			List of shift tuples covering the full hypercube minus the origin.

		"""
		return [
			shift
			for shift in product(range(-self.radius, self.radius + 1), repeat=self.num_spatial_dims)
			if shift != (0,) * self.num_spatial_dims
		]
