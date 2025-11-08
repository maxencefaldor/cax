"""Particle Lenia perceive module."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import State
from cax.core.perceive import Perceive, Perception

from ..lenia.growth import exponential_growth_fn
from .kernel import peak_kernel_fn
from .rule import ParticleLeniaRuleParams


class ParticleLeniaPerceive(Perceive):
	"""Particle Lenia perceive class."""

	def __init__(
		self,
		num_spatial_dims: int,
		*,
		kernel_fn: Callable[[Array, Any], Array] = peak_kernel_fn,
		growth_fn: Callable[[Array, Any], Array] = exponential_growth_fn,
		rule_params: ParticleLeniaRuleParams,
	):
		"""Initialize Particle Lenia perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			kernel_fn: Kernel function.
			growth_fn: Growth mapping function.
			rule_params: Parameters for the rules.

		"""
		self.num_spatial_dims = num_spatial_dims

		self.kernel_fn = kernel_fn
		self.kernel_params = nnx.data(rule_params.kernel_params)

		self.growth_fn = growth_fn
		self.growth_params = nnx.data(rule_params.growth_params)

		self.c_rep = rule_params.c_rep

	def __call__(self, state: State) -> Perception:
		"""Apply Particle Lenia perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state.

		"""
		grad_E = jax.grad(lambda x: self.energy_field(state, x))
		return -jax.vmap(grad_E)(state)

	def compute_fields(self, state: State, x: State) -> tuple[Array, Array, Array]:
		"""Compute energy field."""
		r = jnp.sqrt(jnp.clip(jnp.sum(jnp.square(x - state), axis=-1), min=1e-10))

		# Compute Lenia field
		U = jnp.sum(self.kernel_fn(r, self.kernel_params))

		# Compute growth field
		G = self.growth_fn(U, self.growth_params)

		# Compute repulsion potential field
		R = 0.5 * self.c_rep * jnp.sum(jnp.maximum(1.0 - r, 0.0) ** 2)

		# Return energy field
		return U, G, R

	def energy_field(self, state: State, x: State) -> Array:
		"""Compute energy field."""
		_, G, R = self.compute_fields(state, x)
		return R - G
