"""Lenia perception module for cellular automata."""

import jax.numpy as jnp
from chex import Numeric
from jax import Array

from cax.core.perceive.perceive import Perceive
from cax.types import Perception, State


def bell(x: Array, mean: Numeric, stdev: Numeric) -> Array:
	"""Compute the bell curve (Gaussian function) for given input values.

	Args:
		x: Input values.
		mean: Mean of the bell curve.
		stdev: Standard deviation of the bell curve.

	Returns:
		Bell curve values for the input.

	"""
	return jnp.exp(-(((x - mean) / stdev) ** 2) / 2)


class LeniaPerceive(Perceive):
	"""Lenia perception layer for cellular automata.

	This class implements a perception mechanism using for Lenia.
	"""

	_config: dict
	kernel_fft: Array
	reshape_c_k: Array

	def __init__(self, config: dict):
		"""Initialize the LeniaPerceive layer.

		Args:
			config: Configuration dictionary for Lenia parameters.

		"""
		super().__init__()
		self._config = config
		self.init()

	def init(self) -> None:
		"""Initialize the Lenia kernel and related parameters."""
		kernel_params = self._config["kernel_params"]
		r = self._config["R"] * self._config["state_scale"]

		self.reshape_c_k = jnp.zeros(
			shape=(self._config["channel_size"], len(self._config["kernel_params"]))
		)  # (c, k,)
		for i, kernel in enumerate(kernel_params):
			self.reshape_c_k = self.reshape_c_k.at[kernel["c0"], i].set(1.0)

		# Compute kernel
		mid = self._config["state_size"] // 2
		x = jnp.mgrid[-mid:mid, -mid:mid] / r  # (d, y, x,), coordinates
		d = jnp.linalg.norm(x, axis=0)  # (y, x,), distance from origin
		ds = [d * len(k["b"]) / k["r"] for k in kernel_params]  # (y, x,)*k
		ks = [
			(len(k["b"]) > d)
			* jnp.asarray(k["b"])[jnp.minimum(d.astype(int), len(k["b"]) - 1)]
			* bell(d % 1, 0.5, 0.15)
			for d, k in zip(ds, kernel_params)
		]  # (x, y,)*k
		kernel = jnp.dstack(ks)  # (y, x, k,)
		kernel_normalized = kernel / jnp.sum(kernel, axis=(0, 1), keepdims=True)  # (y, x, k,)
		self.kernel_fft = jnp.fft.fft2(jnp.fft.fftshift(kernel_normalized, axes=(0, 1)), axes=(0, 1))  # (y, x, k,)

	def __call__(self, state: State) -> Perception:
		"""Apply Lenia perception to the input state.

		Args:
			state: Input state of the cellular automaton.

		Returns:
			The perceived state after applying Lenia convolution.

		"""
		state_fft = jnp.fft.fft2(state, axes=(-3, -2))  # (y, x, c,)
		state_fft_k = jnp.dot(state_fft, self.reshape_c_k)  # (y, x, k,)
		u_k = jnp.real(jnp.fft.ifft2(self.kernel_fft * state_fft_k, axes=(-3, -2)))  # (y, x, k,)
		return u_k
