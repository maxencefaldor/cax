"""Particle Life update module.

This module implements the update rule for Particle Life, which integrates interaction
forces to update particle velocities and positions. Includes velocity damping through
friction and periodic boundary conditions.
"""

import jax.numpy as jnp

from cax.core import Input
from cax.core.update import Update

from .perception import ParticleLifePerception
from .state import ParticleLifeState


class ParticleLifeUpdate(Update):
	"""Particle Life update rule.

	Updates particle positions and velocities by integrating interaction forces. Applies
	friction as exponential velocity decay and wraps positions using periodic boundary
	conditions.
	"""

	def __init__(
		self,
		*,
		dt: float = 0.01,
		velocity_half_life: float = 0.01,
	):
		"""Initialize Particle Life update.

		Args:
			dt: Time step of the simulation in arbitrary time units. Smaller values
				produce smoother motion but require more steps for the same duration.
			velocity_half_life: Time constant for velocity decay due to friction. After
				this time, velocity is halved without force input. Smaller values create
				more damped, viscous dynamics.

		"""
		self.dt = dt
		self.friction_factor = float(jnp.power(0.5, dt / velocity_half_life))

	def __call__(
		self,
		state: ParticleLifeState,
		perception: ParticleLifePerception,
		input: Input | None = None,
	) -> ParticleLifeState:
		"""Process the current state, perception, and input to produce a new state.

		Integrates interaction forces to update velocities with friction, then updates
		positions. Applies periodic boundary conditions by wrapping positions to [0, 1].

		Args:
			state: ParticleLifeState containing current class_, position, and velocity arrays
				with shape (num_particles, num_spatial_dims).
			perception: ParticleLifePerception containing acceleration array with shape
				(num_particles, num_spatial_dims) from the perception step.
			input: Optional input (unused in this implementation).

		Returns:
			New ParticleLifeState with updated positions and velocities (class unchanged).

		"""
		velocity = self.friction_factor * state.velocity + self.dt * perception.acceleration
		position = state.position + self.dt * velocity

		# Apply periodic boundary conditions
		position = position % 1.0

		return state.replace(position=position, velocity=velocity)
