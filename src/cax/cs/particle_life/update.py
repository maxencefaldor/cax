"""Particle Life update."""

import jax.numpy as jnp

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class ParticleLifeUpdate(Update):
	"""Particle Life update class."""

	def __init__(
		self,
		*,
		dt: float = 0.01,
		velocity_half_life: float = 0.01,
	):
		"""Initialize Particle Life update.

		Args:
			dt: Time step of the simulation.
			velocity_half_life: Velocity half life for friction.

		"""
		self.dt = dt
		self.friction_factor = float(jnp.power(0.5, dt / velocity_half_life))

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Particle Life rules to update the state.

		Args:
			state: Current state.
			perception: Perceived state, including cell state and neighbor count.
			input: Input (unused in this implementation).

		Returns:
			Next state.

		"""
		velocity = self.friction_factor * state.velocity + self.dt * perception.acceleration
		position = state.position + self.dt * velocity

		# Apply periodic boundary conditions
		position = position % 1.0

		return state.replace(position=position, velocity=velocity)
