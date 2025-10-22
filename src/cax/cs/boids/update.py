"""Boids update module."""

import jax.numpy as jnp

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class BoidsUpdate(Update):
	"""Boids update class."""

	def __init__(
		self,
		*,
		dt: float = 0.01,
		velocity_half_life: float = 0.01,
		boundary: str = "CIRCULAR",
	):
		"""Initialize the Boids Update.

		Args:
			dt: Time step of the simulation.
			velocity_half_life: Velocity half life for friction.
			boundary: Boundary condition.

		"""
		self.dt = dt
		self.friction_factor = float(jnp.power(0.5, dt / velocity_half_life))
		self.boundary = boundary

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Boids rules to update the cellular automaton state.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		velocity = self.friction_factor * state.velocity + perception.acceleration * self.dt
		position = state.position + velocity * self.dt

		# Apply periodic boundary conditions
		if self.boundary == "CIRCULAR":
			position = position % 1.0

		return state.replace(position=position, velocity=velocity)
