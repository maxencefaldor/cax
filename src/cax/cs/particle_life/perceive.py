"""Particle Life perceive module.

This module implements the perception function for Particle Life, which computes pairwise
interaction forces between particles based on their types and distances. Forces transition
from repulsion at short range to attraction at medium range according to the attraction matrix.
"""

import jax.numpy as jnp
from jax import Array

from cax.core import State
from cax.core.perceive import Perceive, Perception

from .perception import ParticleLifePerception as ParticleLifePerception


class ParticleLifePerceive(Perceive):
	"""Particle Life perception.

	Computes interaction forces between all particle pairs based on their distances and
	types. The force profile transitions from repulsion at short distances to attraction
	at medium distances, with interaction strength determined by the attraction matrix.
	"""

	def __init__(
		self,
		*,
		force_factor: float = 1.0,
		r_max: float = 0.15,
		beta: float = 0.3,
		A: Array,
	):
		"""Initialize Particle Life perceive.

		Args:
			force_factor: Global scaling factor for all interaction forces. Higher values
				create stronger, more dynamic interactions.
			r_max: Maximum interaction distance in coordinate space [0, 1]. Particles beyond
				this distance do not interact. Larger values increase computation cost.
			beta: Distance threshold parameter controlling the transition from repulsion to
				attraction. Typically in range [0, 1], where smaller values create stronger
				short-range repulsion.
			A: Attraction matrix of shape (num_classes, num_classes) where A[i, j] defines
				the attraction strength from type i to type j. Positive values attract,
				negative values repel. Values typically range from -1 to 1.

		"""
		self.force_factor = force_factor
		self.r_max = r_max
		self.beta = beta
		self.A = A

	def _get_forces(self, distances: Array, attraction_factors: Array) -> Array:
		"""Calculate interaction forces between particles based on distance.

		Computes forces using a piecewise function: linear repulsion at short distances
		(r <= beta), parameterized attraction at medium distances (beta < r <= r_max),
		and zero force beyond r_max.

		Args:
			distances: Array of normalized pairwise distances (scaled by r_max).
			attraction_factors: Array of attraction coefficients for each particle pair
				from the attraction matrix A.

		Returns:
			Array of scalar force magnitudes with the same shape as distances, where
				positive values indicate repulsion and negative values indicate attraction.

		"""
		distances /= self.r_max
		return jnp.select(
			condlist=[distances <= self.beta, (distances > self.beta) & (distances <= 1)],
			choicelist=[
				distances / self.beta - 1,
				attraction_factors * (1 - jnp.abs(2 * distances - 1 - self.beta) / (1 - self.beta)),
			],
			default=0.0,
		)

	def _get_acceleration(self, forces: Array, direction_norm: Array) -> Array:
		"""Calculate accelerations by summing vectorial forces.

		Converts scalar forces to vector forces by multiplying with direction vectors,
		then sums over all particle pairs to get the total acceleration for each particle.

		Args:
			forces: Array of scalar force magnitudes between particle pairs.
			direction_norm: Array of normalized direction vectors pointing from each particle
				to every other particle.

		Returns:
			Array of acceleration vectors for each particle, with the same shape as positions.

		"""
		return self.force_factor * jnp.sum(forces[..., None] * direction_norm, axis=-2)

	def __call__(self, state: State) -> Perception:
		"""Process the current state to produce a perception.

		Computes pairwise distances between all particles with periodic boundary conditions,
		determines interaction forces based on particle types and the attraction matrix,
		and aggregates forces into acceleration vectors for each particle.

		Args:
			state: ParticleLifeState containing class_, position, and velocity arrays.
				Position should have shape (num_particles, num_spatial_dims).

		Returns:
			ParticleLifePerception containing acceleration array with shape
				(num_particles, num_spatial_dims) representing the total force on each particle.

		"""
		num_particles = state.class_.shape[-1]
		attraction_factors = self.A[state.class_[..., :, None], state.class_[..., None, :]]

		pos_diff = state.position[..., None, :, :] - state.position[..., :, None, :]

		# Apply periodic boundary conditions
		pos_diff = jnp.where(pos_diff > 0.5, pos_diff - 1.0, pos_diff)
		pos_diff = jnp.where(pos_diff < -0.5, pos_diff + 1.0, pos_diff)

		# Calculate distances and normalized directions with periodic conditions
		distance = jnp.linalg.norm(pos_diff, axis=-1)
		direction_norm = jnp.where(
			jnp.eye(num_particles)[..., None], 0.0, pos_diff / distance[..., None]
		)

		# Calculate forces
		forces = self._get_forces(distance, attraction_factors)

		# Calculate accelerations
		acceleration = self._get_acceleration(forces, direction_norm)

		return ParticleLifePerception(acceleration=acceleration)
