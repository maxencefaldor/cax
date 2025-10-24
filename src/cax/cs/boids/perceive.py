"""Boids perceive module."""

import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import Perceive
from cax.types import State

from .perception import BoidsPerception


class BoidsPerceive(Perceive):
	"""Boids perceive class."""

	def __init__(self, boid_policy: nnx.Module):
		"""Initialize Boids perceive.

		Args:
			boid_policy: Boid policy.

		"""
		self.boid_policy = boid_policy

	def __call__(self, state: State) -> BoidsPerception:
		"""Apply Boids perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The Boids perception.

		"""
		num_boids = state.position.shape[-2]

		state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
		acceleration = nnx.split_rngs(splits=num_boids)(
			nnx.vmap(
				lambda boid_policy, state, boid_idx: boid_policy(state, boid_idx),
				in_axes=(state_axes, None, 0),
			)
		)(self.boid_policy, state, jnp.arange(num_boids))

		return BoidsPerception(acceleration=acceleration)
