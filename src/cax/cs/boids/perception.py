"""Boids perception module."""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoidsPerception:
    """Boids perception class."""

    acceleration: Array
