"""Lenia update module."""

import jax.numpy as jnp
from chex import Numeric
from flax import nnx
from jax import Array

from cax.core.perceive.lenia_perceive import bell
from cax.core.update.update import Update
from cax.types import Input, Perception, State


def growth(x: Array, mean: Numeric, stdev: Numeric) -> Array:
	"""Calculate the growth function for Lenia.

	Args:
		x: Input value.
		mean: Mean of the bell curve.
		stdev: Standard deviation of the bell curve.

	Returns:
		The growth value.

	"""
	return 2 * bell(x, mean, stdev) - 1


class LeniaUpdate(Update):
	"""Lenia update class."""

	_config: dict
	reshape_k_c: nnx.Param

	def __init__(self, config: dict) -> None:
		"""Initialize the LeniaUpdate.

		Args:
			config: Configuration dictionary for Lenia parameters.

		"""
		super().__init__()

		self._config = config
		self.init()

	def init(self) -> None:
		"""Initialize Lenia parameters from the configuration."""
		m = jnp.array([k["m"] for k in self._config["kernel_params"]])  # (k,)
		s = jnp.array([k["s"] for k in self._config["kernel_params"]])  # (k,)
		h = jnp.array([k["h"] for k in self._config["kernel_params"]])  # (k,)

		self.m = nnx.Param(m[None, None, ...])  # (1, 1, k,)
		self.s = nnx.Param(s[None, None, ...])  # (1, 1, k,)
		self.h = nnx.Param(h[None, None, ...])  # (1, 1, k,)

		reshape_k_c = jnp.zeros(shape=(len(self._config["kernel_params"]), self._config["channel_size"]))  # (k, c,)
		for i, k in enumerate(self._config["kernel_params"]):
			reshape_k_c = reshape_k_c.at[i, k["c1"]].set(1.0)
		self.reshape_k_c = nnx.Param(reshape_k_c)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Lenia update rule.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state.
			input: External input (unused in this implementation).

		Returns:
			Updated state after applying the Lenia rule.

		"""
		g_k = growth(perception, self.m.value, self.s.value) * self.h.value  # (y, x, k,)
		g = jnp.dot(g_k, self.reshape_k_c.value)  # (y, x, c,)
		state = jnp.clip(state + 1 / self._config["T"] * g, 0.0, 1.0)  # (y, x, c,)
		return state
