"""Reaction-Diffusion module.

This module implements the Gray-Scott reaction-diffusion system, a two-species
continuous cellular automaton that produces a rich variety of Turing patterns
including spots, stripes, mitosis, and coral-like structures. The system models
two chemical species U and V that diffuse at different rates and interact through
a cubic autocatalytic reaction.
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.utils import clip_and_uint8

from .perceive import ReactionDiffusionPerceive
from .update import ReactionDiffusionUpdate


class ReactionDiffusion(ComplexSystem[Array, Array]):
	"""Gray-Scott reaction-diffusion system.

	A continuous cellular automaton modeling two chemical species (U and V) that
	diffuse and react on a grid. The dynamics follow:
		dU/dt = D_u * lap(U) - U*V^2 + f*(1 - U)
		dV/dt = D_v * lap(V) + U*V^2 - (f + k)*V

	Different parameter regimes (feed rate f and kill rate k) produce diverse
	pattern types: spots, stripes, waves, mitosis, and more.
	"""

	def __init__(
		self,
		*,
		num_spatial_dims: int = 2,
		diffusion_rate_u: float = 0.16,
		diffusion_rate_v: float = 0.08,
		feed_rate: float = 0.06,
		kill_rate: float = 0.062,
		dt: float = 1.0,
		rngs: nnx.Rngs,
	):
		"""Initialize Reaction-Diffusion.

		Args:
			num_spatial_dims: Number of spatial dimensions (default 2).
			diffusion_rate_u: Diffusion coefficient for species U.
			diffusion_rate_v: Diffusion coefficient for species V.
			feed_rate: Feed rate f — controls how quickly U is replenished.
			kill_rate: Kill rate k — controls how quickly V is removed.
			dt: Time step size for the Euler integration.
			rngs: RNG key.

		"""
		self.perceive = ReactionDiffusionPerceive(num_spatial_dims=num_spatial_dims, rngs=rngs)
		self.update = ReactionDiffusionUpdate(
			diffusion_rate_u=diffusion_rate_u,
			diffusion_rate_v=diffusion_rate_v,
			feed_rate=feed_rate,
			kill_rate=kill_rate,
			dt=dt,
		)

	def _step(self, state: Array, input: Array | None = None) -> Array:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		return next_state

	@nnx.jit
	def render(self, state: Array) -> Array:
		"""Render state to RGB image.

		Maps the two-species state to an RGB visualization. Species V concentration
		is used as the primary visual signal: high V appears as colored regions against
		a background determined by U.

		Args:
			state: Array with shape (..., *spatial_dims, 2) where channel 0 is U
				concentration and channel 1 is V concentration, both in [0, 1].

		Returns:
			RGB image with dtype uint8 and shape (..., *spatial_dims, 3).

		"""
		u = state[..., 0:1]
		v = state[..., 1:2]

		r = 1.0 - v
		g = 1.0 - 0.5 * v - 0.5 * u
		b = u

		rgb = jnp.concatenate([r, g, b], axis=-1)
		return clip_and_uint8(rgb)
