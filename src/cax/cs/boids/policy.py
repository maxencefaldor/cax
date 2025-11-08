"""Boid policy following Craig W. Reynolds (1987).

References:
	[1] Craig W. Reynolds. 1987. Flocks, herds and schools: A distributed behavioral model.
	[2] https://www.red3d.com/cwr/papers/1987/boids.html

"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .state import BoidsState


class BoidPolicy(nnx.Module):
	"""Boid policy according to Craig Reynolds' paper."""

	def __init__(
		self,
		*,
		acceleration_max: float = jnp.inf,
		acceleration_scale: float = 1.0,
		perception: float = 0.2,
		separation_distance: float = 0.02,
		separation_weight: float = 4.5,
		alignment_weight: float = 0.65,
		cohesion_weight: float = 0.75,
		noise_scale: float = 0.05,
		rngs: nnx.Rngs,
	):
		"""Initialize boid policy.

		Args:
			acceleration_max: Maximum acceleration.
			acceleration_scale: Scale for acceleration.
			perception: Perception radius.
			separation_distance: Separation distance.
			separation_weight: Weight for separation force.
			alignment_weight: Weight for alignment force.
			cohesion_weight: Weight for cohesion force.
			noise_scale: Scale for noise.
			rngs: rng key.

		"""
		self.acceleration_max = acceleration_max
		self.acceleration_scale = acceleration_scale
		self.perception = perception
		self.separation_distance = separation_distance

		self.separation_weight = nnx.Param(separation_weight)
		self.alignment_weight = nnx.Param(alignment_weight)
		self.cohesion_weight = nnx.Param(cohesion_weight)
		self.noise_scale = nnx.Param(noise_scale)

		self.rngs = rngs

	def _toroidal_distance2(self, position_1: Array, position_2: Array) -> Array:
		"""Calculate squared distance considering toroidal world in [0, 1]^n."""
		# Calculate component-wise distances
		vector = self._toroidal_vector(position_1, position_2)

		# Return squared Euclidean distance
		return jnp.sum(vector**2)

	def _toroidal_vector(self, position_1: Array, position_2: Array) -> Array:
		"""Get vector from position_1 to position_2 considering toroidal world in [0, 1]^n."""
		# Calculate component-wise displacements
		pos_diff = position_2 - position_1

		# Apply periodic boundary conditions
		pos_diff = jnp.where(pos_diff > 0.5, pos_diff - 1.0, pos_diff)
		pos_diff = jnp.where(pos_diff < -0.5, pos_diff + 1.0, pos_diff)

		return pos_diff

	def _normalize(self, vector: Array) -> Array:
		"""Normalize a vector."""
		norm = jnp.maximum(jnp.linalg.norm(vector), 1e-8)
		return vector / norm

	def _clip_by_norm(self, vector: Array, max_val: float) -> Array:
		"""Limit the magnitude of a vector."""
		norm = jnp.linalg.norm(vector)
		return jnp.where(norm > max_val, vector * max_val / norm, vector)

	def separation(self, state: BoidsState, boid_idx: int) -> Array:
		"""Calculate separation force for a boid."""
		# Calculate distances to all other boids
		distances = jax.vmap(
			lambda position: self._toroidal_distance2(state.position[boid_idx], position)
		)(state.position)

		# Create masks for filtering
		is_self = jnp.arange(len(state.position)) == boid_idx
		is_too_close = distances <= self.separation_distance**2

		# Only consider other boids that are too close
		separation_mask = ~is_self & is_too_close

		# Calculate steering force
		seperations = -self._toroidal_vector(state.position[boid_idx], state.position)
		steer = jnp.sum(seperations, axis=0, where=separation_mask[..., None])

		return self._normalize(steer)

	def alignment(self, state: BoidsState, boid_idx: int) -> Array:
		"""Calculate alignment force for a boid."""
		# Calculate distances to all other boids
		distances = jax.vmap(
			lambda position: self._toroidal_distance2(state.position[boid_idx], position)
		)(state.position)

		# Create masks for filtering
		is_self = jnp.arange(len(state.position)) == boid_idx
		is_in_perception = distances <= self.perception**2

		# Only consider other boids within perception radius
		perception_mask = ~is_self & is_in_perception

		# Calculate steering force
		velocity_avg = jnp.mean(state.velocity, axis=0, where=perception_mask[..., None])
		steer = velocity_avg - state.velocity[boid_idx]

		return self._normalize(steer)

	def cohesion(self, state: BoidsState, boid_idx: int) -> Array:
		"""Calculate cohesion force for a boid."""
		# Calculate distances to all other boids
		distances = jax.vmap(
			lambda position: self._toroidal_distance2(state.position[boid_idx], position)
		)(state.position)

		# Create masks for filtering
		is_self = jnp.arange(len(state.position)) == boid_idx
		is_in_perception = distances <= self.perception**2

		# Only consider other boids within perception radius
		perception_mask = ~is_self & is_in_perception

		# Calculate steering force
		position_avg = jax.vmap(
			lambda position: self._toroidal_vector(state.position[boid_idx], position)
		)(state.position)
		steer = jnp.mean(position_avg, axis=0, where=perception_mask[..., None])

		return self._normalize(steer)

	def __call__(self, state: BoidsState, boid_idx: int) -> Array:
		"""Apply the boid policy.

		Args:
			state: Position and velocity of all boids.
			boid_idx: Index of the current boid.

		Returns:
			Acceleration of the current boid.

		"""
		# Apply each rule, get resulting forces, and weight them
		separation_update = self.separation_weight * self.separation(state, boid_idx)
		alignment_update = self.alignment_weight * self.alignment(state, boid_idx)
		cohesion_update = self.cohesion_weight * self.cohesion(state, boid_idx)

		# Combine forces
		acceleration = separation_update + alignment_update + cohesion_update

		# Scale and add noise
		acceleration *= self.acceleration_scale
		acceleration += self.noise_scale * jax.random.uniform(
			self.rngs.params(),
			shape=acceleration.shape,
			minval=-1.0,
			maxval=1.0,
		)

		# Limit acceleration
		acceleration = self._clip_by_norm(acceleration, self.acceleration_max)

		return acceleration
