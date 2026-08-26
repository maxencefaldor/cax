"""Pool module."""

from typing import Any, Self

import jax
from flax import nnx
from jax import Array


@nnx.dataclass
class Pool(nnx.Pytree):
	"""A container for PyTree arrays supporting in-place updates and random sampling.

	The pool holds a PyTree of arrays whose first dimension is the pool size. It can be created
	from a PyTree with leading batch dimension. Sampling returns indices and the sliced
	batch for the same indices across all leaves.

	Attributes:
		size: Number of items in the pool (inferred from the leading dimension of the data).
		data: PyTree of arrays stacked along the leading dimension.

	"""

	size: int = nnx.static()
	data: Any = nnx.data()

	@classmethod
	def create(cls, data: Any) -> Self:
		"""Create a new Pool instance.

		Args:
			data: PyTree whose leaves are arrays with shape `(N, ...)`, where `N` is the pool size.

		Returns:
			A new Pool instance with `size == N` and `data == data`.

		"""
		size = jax.tree.leaves(data)[0].shape[0]
		return cls(size=size, data=data)

	@nnx.jit
	def update(self, idxs: Array, batch: Any) -> Self:
		"""Update batch in the pool at the specified indices.

		Args:
			idxs: Integer indices with shape `(B,)` indicating rows to overwrite.
			batch: PyTree matching `data` leaves sliced to `(B, ...)`.

		Returns:
			Pool with the updated batch applied at `idxs` across all leaves.

		"""
		self.data = jax.tree.map(
			lambda data_leaf, batch_leaf: data_leaf.at[idxs].set(batch_leaf), self.data, batch
		)
		return self

	@nnx.jit(static_argnames=("batch_size", "replace"))
	def sample(self, key: Array, *, batch_size: int, replace: bool = True) -> tuple[Array, Any]:
		"""Sample a batch from the pool.

		With `replace=False` the indices are distinct: a later `update(idxs, batch)`
		writes each row exactly once, with no collision between duplicate indices
		deciding which write survives — at the cost of capping `batch_size` at the
		pool size.

		Args:
			key: JAX PRNG key.
			batch_size: Number of rows to sample. With `replace=False`, must not exceed
				the pool size.
			replace: Whether to sample with replacement.

		Returns:
			A tuple `(idxs, batch)` where `idxs` has shape `(batch_size,)` and `batch` is a PyTree
			with each leaf shaped `(batch_size, ...)`.

		"""
		if not replace and batch_size > self.size:
			raise ValueError(f"batch_size ({batch_size}) must not exceed pool size ({self.size})")
		idxs = jax.random.choice(key, self.size, shape=(batch_size,), replace=replace)
		batch = jax.tree.map(lambda leaf: leaf[idxs], self.data)
		return idxs, batch
