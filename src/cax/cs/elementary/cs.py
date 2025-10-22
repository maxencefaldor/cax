"""Elementary Cellular Automata model."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.cs import ComplexSystem, metrics_fn
from cax.types import State
from cax.utils import clip_and_uint8

from .perceive import ElementaryPerceive
from .update import ElementaryUpdate


class Elementary(ComplexSystem):
	"""Elementary Cellular Automata model."""

	def __init__(self, rngs: nnx.Rngs, *, metrics_fn: Callable = metrics_fn):
		"""Initialize Elementary CA."""
		perceive = ElementaryPerceive(rngs=rngs)
		update = ElementaryUpdate(rngs=rngs)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
