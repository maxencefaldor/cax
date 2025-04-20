"""Buffer module."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from cax.types import PyTree


class Buffer(struct.PyTreeNode):
	"""Buffer class."""

	size: int = struct.field(pytree_node=False)
	data: PyTree
	is_full: Array
	idx: Array

	@classmethod
	def create(cls, size: int, datum: PyTree) -> "Buffer":
		"""Create a new Buffer instance.

		Args:
			size: Size of the buffer.
			datum: Unit of data used for inferring the structure of the experience data.

		Returns:
			A new Buffer instance.

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

	@jax.jit
	def add(self, batch: PyTree) -> "Buffer":
		"""Add a batch to the buffer.

		Args:
			batch: A batch to add to the buffer.

		Returns:
			A new Buffer instance with the added batch.

		"""
		batch_size = jax.tree.leaves(batch)[0].shape[0]
		idxs = self.idx + jnp.arange(batch_size)
		idxs = idxs % self.size

		# Update data
		data = jax.tree.map(lambda data, batch: data.at[idxs].set(batch), self.data, batch)

		# Update is_full and idx
		is_full = self.is_full.at[idxs].set(True)
		new_idx = (self.idx + batch_size) % self.size

		return self.replace(data=data, is_full=is_full, idx=new_idx)

	@partial(jax.jit, static_argnames=("batch_size",))
	def sample(self, key: Array, *, batch_size: int) -> PyTree:
		"""Sample a batch from the buffer.

		Args:
			key: A random key.
			batch_size: The size of the batch to sample.

		Returns:
			A batch sampled from the buffer.

		"""
		idxs = jax.random.choice(key, self.size, shape=(batch_size,), p=self.is_full)
		batch: PyTree = jax.tree.map(lambda leaf: leaf[idxs], self.data)
		return batch
