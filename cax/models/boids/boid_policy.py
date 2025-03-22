"""Boid policy following Craig W. Reynolds (1987).

[1] Craig W. Reynolds. 1987. Flocks, herds and schools: A distributed behavioral model.
[2] https://www.red3d.com/cwr/papers/1987/boids.html
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .state import State


class BoidPolicy(nnx.Module):
	"""Boid policy according to Craig Reynolds' paper."""

	def __init__(
		self,
		rngs: nnx.Rngs,
		*,
		separation_weight: float = 4.5,
		alignment_weight: float = 0.65,
		cohesion_weight: float = 0.75,
		perception: float = 0.2,
		separation_distance: float = 0.02,
		acceleration_scale: float = 1.0,
		noise_scale: float = 0.05,
		acceleration_max: float = jnp.inf,
	):
		"""Initialize the boid policy."""
		self.rngs = rngs
		self.separation_weight = separation_weight
		self.alignment_weight = alignment_weight
		self.cohesion_weight = cohesion_weight
		self.perception = perception
		self.separation_distance = separation_distance
		self.acceleration_scale = acceleration_scale
		self.noise_scale = noise_scale
		self.acceleration_max = acceleration_max

	def _toroidal_distance2(self, position_1: jax.Array, position_2: jax.Array) -> jax.Array:
		"""Calculate squared distance considering toroidal world in [0, 1]^n."""
		# Calculate component-wise distances
		vector = self._toroidal_vector(position_1, position_2)

		# Return squared Euclidean distance
		return jnp.sum(vector**2)

	def _toroidal_vector(self, position_1: jax.Array, position_2: jax.Array) -> jax.Array:
		"""Get vector from position_1 to position_2 considering toroidal world in [0, 1]^n."""
		# Calculate component-wise displacements
		pos_diff = position_2 - position_1

		# Apply periodic boundary conditions
		pos_diff = jnp.where(pos_diff > 0.5, pos_diff - 1.0, pos_diff)
		pos_diff = jnp.where(pos_diff < -0.5, pos_diff + 1.0, pos_diff)

		return pos_diff

	def _normalize(self, vector: jax.Array) -> jax.Array:
		"""Normalize a vector."""
		norm = jnp.maximum(jnp.linalg.norm(vector), 1e-8)
		return vector / norm

	def _clip_by_norm(self, vector: jax.Array, max_val: float) -> jax.Array:
		"""Limit the magnitude of a vector."""
		norm = jnp.linalg.norm(vector)
		return jnp.where(norm > max_val, vector * max_val / norm, vector)

	def separation(self, state: State, boid_idx: int) -> jax.Array:
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

	def alignment(self, state: State, boid_idx: int) -> jax.Array:
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

	def cohesion(self, state: State, boid_idx: int) -> jax.Array:
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

	def __call__(self, state: State, boid_idx: int) -> jax.Array:
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
		acceleration += (
			jax.random.uniform(
				self.rngs.params(),
				shape=acceleration.shape,
				minval=-1.0,
				maxval=1.0,
			)
			* self.noise_scale
		)

		# Limit acceleration
		acceleration = self._clip_by_norm(acceleration, self.acceleration_max)

		return acceleration
