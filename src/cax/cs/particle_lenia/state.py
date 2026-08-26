"""Particle Lenia state module."""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParticleLeniaState:
	"""Particle Lenia state class.

	Attributes:
		position: Particle positions with shape (num_particles, num_spatial_dims).

	"""

	position: Array
