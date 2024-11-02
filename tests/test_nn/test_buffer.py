"""Tests for the Buffer class."""

import jax
import jax.numpy as jnp
import pytest
from cax.nn.buffer import Buffer


@pytest.fixture
def sample_datum() -> dict[str, jnp.ndarray]:
	"""Create a sample datum for testing."""
	return {"a": jnp.array([1.0]), "b": jnp.array([2, 3])}


@pytest.fixture
def init_buffer(sample_datum: dict[str, jnp.ndarray]) -> Buffer:
	"""Create an initial buffer for testing."""
	return Buffer.create(size=5, datum=sample_datum)


def test_buffer_create(init_buffer: Buffer, sample_datum: dict[str, jnp.ndarray]) -> None:
	"""Test the creation of a Buffer instance."""
	assert init_buffer.size == 5
	assert init_buffer.index == 0
	assert jnp.all(~init_buffer.is_full)


def test_buffer_add(init_buffer: Buffer) -> None:
	"""Test adding data to the buffer."""
	batch = {"a": jnp.array([[1.0], [2.0]]), "b": jnp.array([[4, 5], [6, 7]])}
	updated_buffer = init_buffer.add(batch)

	assert updated_buffer.index == 2
	assert jnp.array_equal(updated_buffer.is_full, jnp.array([True, True, False, False, False]))
	assert jnp.array_equal(updated_buffer.data["a"][:2], batch["a"])
	assert jnp.array_equal(updated_buffer.data["b"][:2], batch["b"])


def test_buffer_add_wraparound(init_buffer: Buffer) -> None:
	"""Test adding data to the buffer with wraparound."""
	batch1 = {"a": jnp.array([[1.0], [2.0], [3.0]]), "b": jnp.array([[1, 2], [3, 4], [5, 6]])}
	batch2 = {"a": jnp.array([[4.0], [5.0], [6.0]]), "b": jnp.array([[7, 8], [9, 10], [11, 12]])}

	buffer = init_buffer.add(batch1)
	buffer = buffer.add(batch2)

	assert buffer.index == 1
	assert jnp.all(buffer.is_full)
	assert jnp.array_equal(buffer.data["a"], jnp.array([[6.0], [2.0], [3.0], [4.0], [5.0]]))
	assert jnp.array_equal(buffer.data["b"], jnp.array([[11, 12], [3, 4], [5, 6], [7, 8], [9, 10]]))


def test_buffer_sample(init_buffer: Buffer) -> None:
	"""Test sampling from the buffer."""
	batch = {"a": jnp.array([[1.0], [2.0], [3.0]]), "b": jnp.array([[4, 5], [6, 7], [8, 9]])}
	buffer = init_buffer.add(batch)

	key = jax.random.PRNGKey(0)
	sampled_batch = buffer.sample(key, batch_size=2)

	assert isinstance(sampled_batch, dict)
	assert sampled_batch["a"].shape == (2, 1)
	assert sampled_batch["b"].shape == (2, 2)
	assert jnp.all(sampled_batch["a"] != 0)  # Ensure we're not sampling from empty slots
	assert jnp.all(sampled_batch["b"] != 0)
	assert jnp.all(sampled_batch["b"] != 0)
