"""Particle Lenia perceive module.

[1] https://google-research.github.io/self-organising-systems/particle-lenia/
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive import Perceive
from cax.types import Perception, State

from ..lenia.growth import exponential_growth_fn
from .kernel import peak_kernel_fn
from .rule import RuleParams


class ParticleLeniaPerceive(Perceive):
	"""Particle Lenia perception.

	This class implements a perception mechanism using for Lenia.
	"""

	def __init__(
		self,
		num_spatial_dims: int,
		rule_params: RuleParams,
		*,
		kernel_fn: Callable = peak_kernel_fn,
		growth_fn: Callable = exponential_growth_fn,
	):
		"""Initialize the LeniaPerceive layer.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			rule_params: Parameters for the rules.
			kernel_fn: Kernel function.
			growth_fn: Growth mapping function.

		"""
		super().__init__()
		self.num_spatial_dims = num_spatial_dims
		self.kernel_fn = kernel_fn
		self.growth_fn = growth_fn

		# Set rule parameters
		self.rule_params = jax.tree.map(nnx.Param, rule_params)

	def __call__(self, state: State) -> Perception:
		"""Apply Particle Lenia perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state.

		"""
		grad_E = jax.grad(lambda x: self.energy_field(state, x))
		return -jax.vmap(grad_E)(state)

	def compute_fields(self, state: State, x: State) -> Array:
		"""Compute energy field."""
		r = jnp.sqrt(jnp.clip(jnp.sum(jnp.square(x - state), axis=-1), min=1e-10))

		# Compute Lenia field
		U = jnp.sum(self.kernel_fn(r, self.rule_params.kernel_params))

		# Compute growth field
		G = self.growth_fn(U, self.rule_params.growth_params)

		# Compute repulsion potential field
		R = 0.5 * self.rule_params.c_rep * jnp.sum(jnp.maximum(1.0 - r, 0.0) ** 2)

		# Return energy field
		return U, G, R

	def energy_field(self, state: State, x: State) -> Array:
		"""Compute energy field."""
		_, G, R = self.compute_fields(state, x)
		return R - G
