"""Particle Life update module."""

import jax.numpy as jnp
from jax import Array

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class ParticleLifeUpdate(Update):
	"""Life update class."""

	def __init__(
		self,
		dt: float = 0.01,
		velocity_half_life: float = 0.01,
		force_factor: float = 1.0,
		boundary: str = "CIRCULAR",
	):
		"""Initialize the Particle Life Update.

		Args:
			force_factor: Force factor.
			boundary: Boundary condition.

		"""
		self.dt = dt
		self.friction_factor = float(jnp.power(0.5, dt / velocity_half_life))
		self.force_factor = force_factor
		self.boundary = boundary

	def get_acceleration(self, forces: Array, direction_norm: Array) -> Array:
		"""Calculate accelerations from forces and direction norms.

		Args:
			forces: Forces acting on particles.
			direction_norm: Normalized direction vectors.

		Returns:
			Accelerations of particles.

		"""
		return self.force_factor * jnp.sum(forces[..., None] * direction_norm, axis=-2)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Game of Life rules to update the cellular automaton state.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		class_, position, velocity = state
		direction_norm, forces = perception

		acceleration = self.get_acceleration(forces, direction_norm)
		velocity *= self.friction_factor

		velocity += acceleration * self.dt
		position += velocity * self.dt

		# Apply periodic boundary conditions
		if self.boundary == "CIRCULAR":
			position = position % 1.0

		return class_, position, velocity
