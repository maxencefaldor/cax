"""Particle Lenia update module.

This module implements the update rule for Particle Lenia, which integrates forces to
update particle positions with temporal resolution T.
"""

from dataclasses import replace

from jax import Array

from cax.core.update import Update

from .state import ParticleLeniaState


class ParticleLeniaUpdate(Update[ParticleLeniaState, Array, Array]):
	"""Particle Lenia update rule.

	Updates particle positions by integrating forces derived from energy field gradients.
	The temporal resolution T controls the step size, with higher T producing smaller,
	smoother position updates.
	"""

	def __init__(
		self,
		*,
		T: float,
	):
		"""Initialize Particle Lenia update.

		Args:
			T: Time resolution controlling the temporal discretization. Higher values
				produce smoother temporal dynamics with smaller update steps.

		"""
		self.T = T

	def __call__(
		self, state: ParticleLeniaState, perception: Array, input: Array | None = None
	) -> ParticleLeniaState:
		"""Process the current state, perception, and input to produce a new state.

		Integrates forces to update particle positions with temporal resolution T. Particles
		move according to the gradient of the energy field, scaled by 1/T for smooth dynamics.

		Args:
			state: ParticleLeniaState containing current particle positions.
			perception: Array with shape (num_particles, num_spatial_dims) containing force
				vectors from the perception step.
			input: Optional input (unused in this implementation).

		Returns:
			Next ParticleLeniaState after position update.

		"""
		return replace(state, position=state.position + perception / self.T)
