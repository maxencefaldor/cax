"""Boids perceive module.

This module implements the perception function for Boids, which computes steering
accelerations for each boid based on its neighbors. The boid policy defines the specific
steering behaviors (separation, alignment, cohesion) and their parameters.

"""

import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import Perceive

from .perception import BoidsPerception
from .state import BoidsState


class BoidsPerceive(Perceive):
	"""Boids perception.

	Computes steering accelerations for each boid by applying the boid policy. The policy
	evaluates each boid's local neighborhood and determines the desired steering direction
	to achieve flocking behaviors.
	"""

	def __init__(self, boid_policy: nnx.Module):
		"""Initialize Boids perceive.

		Args:
			boid_policy: Policy defining the behavior of the boids.

		"""
		self.boid_policy = boid_policy

	def __call__(self, state: BoidsState) -> BoidsPerception:
		"""Process the current state to produce a perception.

		Computes the steering acceleration for each boid by applying the boid policy,
		which evaluates neighbors and determines the desired steering based on separation,
		alignment, and cohesion rules.

		Args:
			state: BoidsState containing position and velocity arrays with shape
				(num_boids, num_spatial_dims).

		Returns:
			BoidsPerception containing acceleration array with shape (num_boids, num_spatial_dims)
				representing the steering forces for each boid.

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
