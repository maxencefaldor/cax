"""Boids Perceive module."""

import jax
import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import Perceive
from cax.types import State

from .perception import Perception


class BoidsPerceive(Perceive):
	"""Boids Perceive class.

	This class implements perception based on the Boids.
	"""

	def __init__(self, boid_policy: nnx.Module):
		"""Initialize the Boids Perceive.

		Args:
			boid_policy: Boid policy.

		"""
		self.boid_policy = boid_policy

	def __call__(self, state: State) -> Perception:
		"""Apply Boids perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The Boids perception.

		"""
		acceleration = jax.vmap(self.boid_policy, in_axes=(None, 0))(
			state, jnp.arange(state.position.shape[-2])
		)

		return Perception(acceleration=acceleration)
