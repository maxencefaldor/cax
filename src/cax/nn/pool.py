"""Pool module."""

from functools import partial

import jax
from flax import struct
from jax import Array

from cax.types import PyTree


class Pool(struct.PyTreeNode):
	"""A container for PyTree arrays supporting in-place updates and random sampling.

	The pool holds a PyTree of arrays whose first dimension is the pool size. It can be created
	from a PyTree with leading batch dimension. Sampling returns indices and the sliced
	batch for the same indices across all leaves.

	Attributes:
		size: Number of items in the pool (inferred from the leading dimension of the data).
		data: PyTree of arrays stacked along the leading dimension.

	"""

	size: int = struct.field(pytree_node=False)
	data: PyTree

	@classmethod
	def create(cls, data: PyTree) -> "Pool":
		"""Create a new Pool instance.

		Args:
			data: PyTree whose leaves are arrays with shape `(N, ...)`, where `N` is the pool size.

		Returns:
			A new Pool instance with `size == N` and `data == data`.

		"""
		size = jax.tree.leaves(data)[0].shape[0]
		return cls(size=size, data=data)

	@jax.jit
	def update(self, idxs: Array, batch: PyTree) -> "Pool":
		"""Update batch in the pool at the specified indices.

		Args:
			idxs: Integer indices with shape `(B,)` indicating rows to overwrite.
			batch: PyTree matching `data` leaves sliced to `(B, ...)`.

		Returns:
			A new Pool instance with the updated batch applied at `idxs` across all leaves.

		"""
		data = jax.tree.map(
			lambda data_leaf, batch_leaf: data_leaf.at[idxs].set(batch_leaf), self.data, batch
		)
		return self.replace(data=data)

	@partial(jax.jit, static_argnames=("batch_size",))
	def sample(self, key: Array, *, batch_size: int) -> tuple[Array, PyTree]:
		"""Sample a batch from the pool.

		Args:
			key: JAX PRNG key.
			batch_size: Number of rows to sample.

		Returns:
			A tuple `(idxs, batch)` where `idxs` has shape `(batch_size,)` and `batch` is a PyTree
			with each leaf shaped `(batch_size, ...)`.

		"""
		idxs = jax.random.choice(key, self.size, shape=(batch_size,))
		batch = jax.tree.map(lambda leaf: leaf[idxs], self.data)
		return idxs, batch
