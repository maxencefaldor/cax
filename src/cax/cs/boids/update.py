"""Boids update module."""

import jax.numpy as jnp

from cax.core import Input
from cax.core.update import Update

from .perception import BoidsPerception
from .state import BoidsState


class BoidsUpdate(Update):
	"""Boids update class."""

	def __init__(
		self,
		*,
		dt: float = 0.01,
		velocity_half_life: float = jnp.inf,
	):
		"""Initialize Boids update.

		Args:
			dt: Time step of the simulation.
			velocity_half_life: Velocity half life for friction.

		"""
		self.dt = dt
		self.friction_factor = float(jnp.power(0.5, dt / velocity_half_life))

	def __call__(
		self, state: BoidsState, perception: BoidsPerception, input: Input | None = None
	) -> BoidsState:
		"""Apply the Boids rules to update the state.

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
