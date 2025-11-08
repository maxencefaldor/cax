"""Boids update module.

This module implements the update rule for Boids, which integrates steering accelerations
to update velocities and positions. Includes optional velocity damping through friction
and periodic boundary conditions.

"""

import jax.numpy as jnp

from cax.core import Input
from cax.core.update import Update

from .perception import BoidsPerception
from .state import BoidsState


class BoidsUpdate(Update):
	"""Boids update rule.

	Updates boid positions and velocities by integrating steering accelerations. Applies
	friction as exponential velocity decay and wraps positions using periodic boundary
	conditions.

	"""

	def __init__(
		self,
		*,
		dt: float = 0.01,
		velocity_half_life: float = jnp.inf,
	):
		"""Initialize Boids update.

		Args:
			dt: Time step of the simulation in arbitrary time units. Smaller values
				produce smoother motion but require more steps for the same duration.
			velocity_half_life: Time constant for velocity decay due to friction. After
				this time, velocity is halved without steering input. Use jnp.inf for no
				friction. Smaller values create more damped, sluggish motion.

		"""
		self.dt = dt
		self.friction_factor = float(jnp.power(0.5, dt / velocity_half_life))

	def __call__(
		self, state: BoidsState, perception: BoidsPerception, input: Input | None = None
	) -> BoidsState:
		"""Process the current state, perception, and input to produce a new state.

		Integrates steering accelerations to update velocities with friction, then updates
		positions. Applies periodic boundary conditions by wrapping positions to [0, 1].

		Args:
			state: BoidsState containing current position and velocity arrays with shape
				(num_boids, num_spatial_dims).
			perception: BoidsPerception containing acceleration array with shape
				(num_boids, num_spatial_dims) from the perception step.
			input: Optional input (unused in this implementation).

		Returns:
			New BoidsState with updated positions and velocities.

		"""
		velocity = self.friction_factor * state.velocity + self.dt * perception.acceleration
		position = state.position + self.dt * velocity

		# Apply periodic boundary conditions
		position = position % 1.0

		return state.replace(position=position, velocity=velocity)
