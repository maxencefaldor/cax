"""Tests for the Pool class."""

import jax
import jax.numpy as jnp
import pytest
from cax.nn.pool import Pool


@pytest.fixture
def init_pool():
	"""Create a Pool instance for testing."""
	return Pool.create({"a": jnp.array([1, 2, 3, 4, 5]), "b": jnp.array([10, 20, 30, 40, 50])})


def test_pool_create(init_pool):
	"""Test the create method of the Pool class."""
	assert init_pool.size == 5
	assert "a" in init_pool.data
	assert "b" in init_pool.data
	assert jnp.array_equal(init_pool.data["a"], jnp.array([1, 2, 3, 4, 5]))
	assert jnp.array_equal(init_pool.data["b"], jnp.array([10, 20, 30, 40, 50]))


def test_pool_sample(init_pool):
	"""Test the sample method of the Pool class."""
	key = jax.random.PRNGKey(0)
	index, sampled_data = init_pool.sample(key, batch_size=3)

	assert index.shape == (3,)
	assert "a" in sampled_data
	assert "b" in sampled_data
	assert sampled_data["a"].shape == (3,)
	assert sampled_data["b"].shape == (3,)


def test_pool_update(init_pool):
	"""Test the update method of the Pool class."""
	new_index = jnp.array([0, 2, 4])
	new_a = jnp.array([100, 200, 300])
	new_b = jnp.array([1000, 2000, 3000])

	updated_pool = init_pool.update(new_index, {"a": new_a, "b": new_b})

	assert jnp.array_equal(updated_pool.data["a"], jnp.array([100, 2, 200, 4, 300]))
	assert jnp.array_equal(updated_pool.data["b"], jnp.array([1000, 20, 2000, 40, 3000]))
