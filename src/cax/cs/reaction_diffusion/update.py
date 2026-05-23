"""Reaction-Diffusion update module.

This module implements the Gray-Scott update rule, which integrates diffusion (from the
Laplacian perception) with the cubic reaction term and feed/kill dynamics for two chemical
species U and V.
"""

import jax.numpy as jnp
from jax import Array

from cax.core.perceive.perceive import Perception
from cax.core.update import Update


class ReactionDiffusionUpdate(Update[Array, Array]):
	"""Gray-Scott reaction-diffusion update rule.

	Applies the Gray-Scott equations in discrete time:
		U' = U + (D_u * lap(U) - U*V^2 + f*(1 - U)) * dt
		V' = V + (D_v * lap(V) + U*V^2 - (f + k)*V) * dt

	where D_u and D_v are diffusion rates, f is the feed rate, k is the kill rate,
	and dt is the time step size.
	"""

	def __init__(
		self,
		*,
		diffusion_rate_u: float = 0.16,
		diffusion_rate_v: float = 0.08,
		feed_rate: float = 0.06,
		kill_rate: float = 0.062,
		dt: float = 1.0,
	):
		"""Initialize Reaction-Diffusion update.

		Args:
			diffusion_rate_u: Diffusion coefficient for species U.
			diffusion_rate_v: Diffusion coefficient for species V.
			feed_rate: Feed rate f — controls how quickly U is replenished.
			kill_rate: Kill rate k — controls how quickly V is removed.
			dt: Time step size for the Euler integration.

		"""
		self.diffusion_rate_u = diffusion_rate_u
		self.diffusion_rate_v = diffusion_rate_v
		self.feed_rate = feed_rate
		self.kill_rate = kill_rate
		self.dt = dt

	def __call__(self, state: Array, perception: Perception, input: Array | None = None) -> Array:
		"""Process the current state, perception, and input to produce a new state.

		Applies one Euler step of the Gray-Scott equations using the Laplacian computed
		by the perception module.

		Args:
			state: Current state (unused, values come from perception).
			perception: Array with shape (..., *spatial_dims, 4) where channels are
				[U, lap(U), V, lap(V)].
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (..., *spatial_dims, 2) containing updated [U, V]
				concentrations clipped to [0, 1].

		"""
		u = perception[..., 0:1]
		lap_u = perception[..., 1:2]
		v = perception[..., 2:3]
		lap_v = perception[..., 3:4]

		reaction = u * v**2

		du = self.diffusion_rate_u * lap_u - reaction + self.feed_rate * (1.0 - u)
		dv = self.diffusion_rate_v * lap_v + reaction - (self.feed_rate + self.kill_rate) * v

		new_u = u + du * self.dt
		new_v = v + dv * self.dt

		state = jnp.concatenate([new_u, new_v], axis=-1)
		return jnp.clip(state, 0.0, 1.0)
