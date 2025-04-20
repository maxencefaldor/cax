"""Particle Life perceive."""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive import Perceive
from cax.types import Perception, State

from .perception import Perception as ParticleLifePerception


class ParticleLifePerceive(Perceive):
	"""Particle Life Perceive class.

	This class implements perception based on the Particle Life.
	"""

	def __init__(
		self,
		A: Array,
		*,
		r_max: float = 0.15,
		beta: float = 0.3,
		force_factor: float = 1.0,
		boundary: str = "CIRCULAR",
	):
		"""Initialize the Particle Life Perceive.

		Args:
			A: Attraction matrix.
			r_max: Maximum distance for attraction.
			beta: Attraction threshold.
			force_factor: Force factor.
			boundary: Boundary condition.

		"""
		self.A = nnx.Param(A)
		self.r_max = r_max
		self.beta = beta
		self.force_factor = force_factor
		self.boundary = boundary

	def get_forces(self, distances: Array, attraction_factors: Array) -> Array:
		"""Calculate interaction forces between particles.

		Args:
			distances: Pairwise distances between particles.
			attraction_factors: Matrix of attraction coefficients between particle types.

		Returns:
			Array of forces between particles.

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

	def get_acceleration(self, forces: Array, direction_norm: Array) -> Array:
		"""Calculate accelerations from forces and direction norms.

		Args:
			forces: Forces acting on particles.
			direction_norm: Normalized direction vectors.

		Returns:
			Accelerations of particles.

		"""
		return self.force_factor * jnp.sum(forces[..., None] * direction_norm, axis=-2)

	def __call__(self, state: State) -> Perception:
		"""Apply Particle Life perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The Particle Life perception.

		"""
		num_particles = state.class_.shape[-1]
		attraction_factors = self.A[state.class_[..., :, None], state.class_[..., None, :]]

		pos_diff = state.position[..., None, :, :] - state.position[..., :, None, :]

		# Apply periodic boundary conditions
		if self.boundary == "CIRCULAR":
			pos_diff = jnp.where(pos_diff > 0.5, pos_diff - 1.0, pos_diff)
			pos_diff = jnp.where(pos_diff < -0.5, pos_diff + 1.0, pos_diff)

		# Calculate distances and normalized directions with periodic conditions
		distance = jnp.linalg.norm(pos_diff, axis=-1)
		direction_norm = jnp.where(
			jnp.eye(num_particles)[..., None], 0.0, pos_diff / distance[..., None]
		)

		# Calculate forces
		forces = self.get_forces(distance, attraction_factors)

		# Calculate accelerations
		acceleration = self.get_acceleration(forces, direction_norm)

		return ParticleLifePerception(acceleration=acceleration)
