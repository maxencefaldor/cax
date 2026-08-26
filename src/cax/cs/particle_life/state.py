"""Particle Life state module."""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParticleLifeState:
	"""Particle Life state class."""

	class_: Array
	position: Array
	velocity: Array
