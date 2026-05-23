"""Buffer module."""

from typing import Self

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.types import PyTree


class Buffer(nnx.Pytree):
	"""A container for PyTree arrays with circular writes and random sampling.

	The buffer stores a PyTree of arrays with a fixed capacity along the leading dimension.
	New batches are written sequentially with wrap-around semantics. Sampling draws indices
	from the subset of entries that have been written at least once.

	Attributes:
		size: Maximum number of items stored.
		data: PyTree of arrays with leading dimension `size`.
		is_full: Boolean mask of shape `(size,)` indicating which entries are initialized.
		idx: Current write pointer (modulo `size`).

	"""

	def __init__(self, size: int, data: PyTree, is_full: Array, idx: Array):
		"""Initialize buffer.

		Args:
			size: Maximum number of items stored.
			data: PyTree of arrays with leading dimension `size`.
			is_full: Boolean mask of shape `(size,)` indicating which entries are initialized.
			idx: Current write pointer (modulo `size`).

		"""
		self.size = size
		self.data = nnx.data(data)
		self.is_full = is_full
		self.idx = idx

	@classmethod
	def create(cls, size: int, datum: PyTree) -> Self:
		"""Create a new Buffer instance.

		Args:
			size: Size of the buffer.
			datum: PyTree example whose leaf dtypes/shapes are used to allocate storage.

		Returns:
			A new Buffer instance with empty storage of capacity `size`.

		"""
		data = jax.tree.map(jnp.empty_like, datum)
		data = jax.tree.map(
			lambda leaf: jnp.broadcast_to(leaf[None, ...], (size, *leaf.shape)), data
		)
		return cls(
			size=size,
			data=data,
			is_full=jnp.full((size,), False, dtype=jnp.bool),
			idx=jnp.array(0, dtype=jnp.int32),
		)

	@nnx.jit
	def add(self, batch: PyTree) -> Self:
		"""Add a batch to the buffer.

		Args:
			batch: PyTree whose leaves have shape `(B, ...)`, where `B` is the batch size.

		Returns:
			Buffer with the batch written at consecutive indices (with wrap-around).

		"""
		batch_size = jax.tree.leaves(batch)[0].shape[0]
		idxs = self.idx + jnp.arange(batch_size)
		idxs = idxs % self.size

		self.data = jax.tree.map(lambda data, batch: data.at[idxs].set(batch), self.data, batch)
		self.is_full = self.is_full.at[idxs].set(True)
		self.idx = (self.idx + batch_size) % self.size
		return self

	@nnx.jit(static_argnames=("batch_size",))
	def sample(self, key: Array, *, batch_size: int) -> PyTree:
		"""Sample a batch from the buffer.

		Args:
			key: JAX PRNG key.
			batch_size: Number of rows to sample from initialized entries.

		Returns:
			A PyTree with each leaf shaped `(batch_size, ...)`, sampled from filled slots.

		"""
		idxs = jax.random.choice(key, self.size, shape=(batch_size,), p=self.is_full)
		batch: PyTree = jax.tree.map(lambda leaf: leaf[idxs], self.data)
		return batch
