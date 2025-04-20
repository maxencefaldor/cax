"""Tests for the Pool class."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from cax.nn.pool import Pool


@pytest.fixture
def data() -> dict[str, Array]:
	"""Create data for testing."""
	return {
		"a": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
		"b": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
	}


@pytest.fixture
def pool(data: dict[str, Array]) -> Pool:
	"""Create a Pool for testing."""
	return Pool.create(data)


def test_pool_create(pool: Pool) -> None:
	"""Test the create method of the Pool class."""
	assert pool.size == 5
	assert jax.tree.reduce(
		lambda x, y: jnp.sum(x) + jnp.sum(y), pool.data, initializer=jnp.array(0.0)
	) == jnp.array(165.0)


def test_pool_sample(pool: Pool) -> None:
	"""Test the sample method of the Pool class."""
	key = jax.random.key(0)
	idxs, sampled_batch = pool.sample(key, batch_size=3)

	assert idxs.dtype == jnp.int32
	assert idxs.shape == (3,)
	assert "a" in sampled_batch
	assert "b" in sampled_batch
	assert sampled_batch["a"].shape == (3,)
	assert sampled_batch["b"].shape == (3,)


def test_pool_update(pool: Pool) -> None:
	"""Test the update method of the Pool class."""
	new_idxs = jnp.array([0, 2, 4])
	new_a = jnp.array([100, 200, 300])
	new_b = jnp.array([1000, 2000, 3000])

	updated_pool = pool.update(new_idxs, {"a": new_a, "b": new_b})

	assert jnp.array_equal(updated_pool.data["a"], jnp.array([100, 2, 200, 4, 300]))
	assert jnp.array_equal(updated_pool.data["b"], jnp.array([1000, 20, 2000, 40, 3000]))
