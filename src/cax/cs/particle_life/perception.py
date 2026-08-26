"""Particle Life perception module."""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParticleLifePerception:
	"""Particle Life perception class."""

	acceleration: Array
