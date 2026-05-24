"""Sandpile perceive module.

This module implements the perception function for the Abelian Sandpile model.
It gathers the chip counts of face-adjacent neighbors needed for the toppling rule,
supporting both periodic (circular) and open (dissipative) boundary conditions.
"""

from typing import Literal

from cax.core.perceive import VonNeumannPerceive


class SandpilePerceive(VonNeumannPerceive):
	"""Sandpile perception.

	Gathers the Von Neumann neighborhood (face-adjacent cells) for each cell.
	In 2D, this produces 5 channels: [self, N, S, E, W] (center + 4 neighbors).

	Supports two boundary modes:
		- "CIRCULAR": periodic boundaries (sand wraps around edges).
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
		neighborhood_padding = "CIRCULAR" if padding == "CIRCULAR" else "ZERO"
		super().__init__(num_spatial_dims=num_spatial_dims, radius=1, padding=neighborhood_padding)
