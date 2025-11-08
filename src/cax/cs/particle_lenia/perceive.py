"""Particle Lenia perceive module.

This module implements the perception function for Particle Lenia, which computes forces
on particles from energy field gradients. The energy field is derived from kernel and
growth fields computed between particle pairs, plus a repulsion term.
"""

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
	"""Particle Lenia perception.

	Computes forces on particles by taking the gradient of an energy field. The energy
	field combines a growth term (derived from kernel and growth functions applied to
	particle pairs) with a repulsion term that prevents particle overlap.
	"""

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
			num_spatial_dims: Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
				Determines the dimensionality of particle positions and field computations.
			kernel_fn: Callable that computes pairwise kernel weights between particles
				based on their distance. Takes rule parameters and returns kernel values.
			growth_fn: Callable that maps kernel field values to growth field values.
				Defines how particles respond to their local neighborhood density.
			rule_params: Instance of ParticleLeniaRuleParams containing kernel and growth
				parameters such as radii, peak positions, widths, and heights.

		"""
		self.num_spatial_dims = num_spatial_dims

		self.kernel_fn = kernel_fn
		self.kernel_params = nnx.data(rule_params.kernel_params)

		self.growth_fn = growth_fn
		self.growth_params = nnx.data(rule_params.growth_params)

		self.c_rep = rule_params.c_rep

	def __call__(self, state: State) -> Perception:
		"""Process the current state to produce a perception.

		Computes the force on each particle by taking the negative gradient of the energy
		field with respect to particle position. Particles are driven toward regions of
		high growth and away from regions of high repulsion.

		Args:
			state: Array with shape (num_particles, num_spatial_dims) containing particle
				positions in continuous space.

		Returns:
			Array with shape (num_particles, num_spatial_dims) containing force vectors
				for each particle.

		"""
		grad_E = jax.grad(lambda x: self.energy_field(state, x))
		return -jax.vmap(grad_E)(state)

	def compute_fields(self, state: State, x: State) -> tuple[Array, Array, Array]:
		"""Compute kernel, growth, and repulsion fields at a position.

		Evaluates the kernel field (neighborhood density), growth field (desirability),
		and repulsion field (overlap prevention) at position x given all particle positions.

		Args:
			state: Array with shape (num_particles, num_spatial_dims) containing all
				particle positions.
			x: Array with shape (num_spatial_dims,) specifying the query position.

		Returns:
			Tuple of (U, G, R) where U is the kernel field value, G is the growth field
				value, and R is the repulsion field value at position x.

		"""
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
		"""Compute energy field at a position.

		The energy field combines repulsion (positive, increases near particles) and
		growth (negative, represents attractiveness) to create a potential landscape
		that drives particle motion.

		Args:
			state: Array with shape (num_particles, num_spatial_dims) containing all
				particle positions.
			x: Array with shape (num_spatial_dims,) specifying the query position.

		Returns:
			Scalar energy value at position x, computed as repulsion minus growth (R - G).

		"""
		_, G, R = self.compute_fields(state, x)
		return R - G
