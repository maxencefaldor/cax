"""Pool implementation for NCA training."""

from functools import partial
from typing import Self

import jax
from flax import struct


class Pool(struct.PyTreeNode):
	"""Pool class for for NCA training."""

	size: int = struct.field(pytree_node=False)
	data: dict[str, jax.Array]

	@classmethod
	def create(cls, **kwargs):
		"""Create a new Pool instance from the given data.

		Args:
			**kwargs: Keyword arguments representing the data to be stored in the pool.

		Returns:
			A new Pool instance.

		"""
		size = next(iter(kwargs.values())).shape[0]
		return cls(size=size, data=kwargs)

	@partial(jax.jit, static_argnames=("sample_size",))
	def sample(self, key: jax.Array, *, sample_size: int) -> tuple[jax.Array, dict[str, jax.Array]]:
		"""Sample a subset of data from the pool.

		Args:
			key: A JAX random key.
			sample_size: The number of items to sample.

		Returns:
			A tuple containing the sampled indices and the sampled data.

		"""
		index = jax.random.choice(key, self.size, shape=(sample_size,), replace=False)
		sampled_data = {k: v[index] for k, v in self.data.items()}
		return index, sampled_data

	@jax.jit
	def add(self, index: jax.Array, **kwargs) -> Self:
		"""Add items in the pool at the specified indices.

		Args:
			index: The indices at which to add or update items.
			**kwargs: The data to be added or updated.

		Returns:
			A new Pool instance with the updated data.

		"""
		updated_data = {k: self.data[k].at[index].set(kwargs[k]) for k in self.data}
		return self.replace(data=updated_data)
