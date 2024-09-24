"""Pool module."""

from functools import partial

import jax
from chex import ArrayTree
from flax import struct
from jax import Array


class Pool(struct.PyTreeNode):
	"""Pool class."""

	size: int = struct.field(pytree_node=False)
	data: ArrayTree

	@classmethod
	def create(cls, data: ArrayTree) -> "Pool":
		"""Create a new Pool instance.

		Args:
			data: Data to store in the pool.

		Returns:
			A new Pool instance.

		"""
		size = jax.tree.leaves(data)[0].shape[0]
		return cls(size=size, data=data)

	@jax.jit
	def update(self, indices: Array, batch: ArrayTree) -> "Pool":
		"""Update batch in the pool at the specified indices.

		Args:
			indices: The indices at which to update the batch.
			batch: The batch to update at the specified indices.

		Returns:
			A new Pool instance with the updated batch.

		"""
		data = jax.tree.map(lambda data_leaf, batch_leaf: data_leaf.at[indices].set(batch_leaf), self.data, batch)
		return self.replace(data=data)

	@partial(jax.jit, static_argnames=("batch_size",))
	def sample(self, key: Array, *, batch_size: int) -> tuple[Array, ArrayTree]:
		"""Sample a batch from the pool.

		Args:
			key: A random key.
			batch_size: The size of the batch to sample.

		Returns:
			A tuple containing the batch indices in the pool and the batch.

		"""
		indices = jax.random.choice(key, self.size, shape=(batch_size,))
		batch = jax.tree.map(lambda leaf: leaf[indices], self.data)
		return indices, batch
