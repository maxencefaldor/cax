"""Boids state module."""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoidsState:
	"""Boids state class."""

	position: Array
	velocity: Array
