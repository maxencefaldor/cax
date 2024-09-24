"""Buffer module."""

from functools import partial

import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax import struct
from jax import Array


class Buffer(struct.PyTreeNode):
	"""Buffer class."""

	size: int = struct.field(pytree_node=False)
	data: ArrayTree
	is_full: Array
	index: Array

	@classmethod
	def create(cls, size: int, datum: ArrayTree) -> "Buffer":
		"""Create a new Buffer instance.

		Args:
			size: Size of the buffer.
			datum: Unit of data used for inferring the structure of the experience data.

		Returns:
			A new Buffer instance.

		"""
		data = jax.tree.map(jnp.empty_like, datum)
		data = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf[None, ...], (size, *leaf.shape)), data)
		return cls(
			size=size,
			data=data,
			is_full=jnp.full((size,), False, dtype=jnp.bool),
			index=jnp.array(0, dtype=jnp.int32),
		)

	@jax.jit
	def add(self, batch: ArrayTree) -> "Buffer":
		"""Add a batch to the buffer.

		Args:
			batch: A batch to add to the buffer.

		Returns:
			A new Buffer instance with the added batch.

		"""
		batch_size = jax.tree.leaves(batch)[0].shape[0]
		indices = self.index + jnp.arange(batch_size)
		indices = indices % self.size

		# Update data
		data = jax.tree.map(lambda data, batch: data.at[indices].set(batch), self.data, batch)

		# Update is_full and index
		is_full = self.is_full.at[indices].set(True)
		new_index = (self.index + batch_size) % self.size

		return self.replace(data=data, is_full=is_full, index=new_index)

	@partial(jax.jit, static_argnames=("batch_size",))
	def sample(self, key: Array, *, batch_size: int) -> ArrayTree:
		"""Sample a batch from the buffer.

		Args:
			key: A random key.
			batch_size: The size of the batch to sample.

		Returns:
			A batch sampled from the buffer.

		"""
		indices = jax.random.choice(key, self.size, shape=(batch_size,), p=self.is_full)
		batch: ArrayTree = jax.tree.map(lambda leaf: leaf[indices], self.data)
		return batch
