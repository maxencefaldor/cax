"""Tests for the Buffer class."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from cax.nn.buffer import Buffer


@pytest.fixture
def datum() -> dict[str, Array]:
	"""Create datum for testing."""
	return {"a": jnp.array([1.0]), "b": jnp.array([2, 3])}


@pytest.fixture
def buffer(datum: dict[str, Array]) -> Buffer:
	"""Create a buffer for testing."""
	return Buffer.create(size=5, datum=datum)


def test_buffer_create(buffer: Buffer) -> None:
	"""Test the creation of a Buffer instance."""
	assert buffer.size == 5
	assert buffer.idx == 0
	assert jnp.all(~buffer.is_full)


def test_buffer_add(buffer: Buffer) -> None:
	"""Test adding data to the buffer."""
	batch = {"a": jnp.array([[1.0], [2.0]]), "b": jnp.array([[4, 5], [6, 7]])}

	new_buffer = buffer.add(batch)

	assert new_buffer.idx == 2
	assert jnp.array_equal(new_buffer.is_full, jnp.array([True, True, False, False, False]))
	assert jnp.array_equal(new_buffer.data["a"][:2], batch["a"])
	assert jnp.array_equal(new_buffer.data["b"][:2], batch["b"])


def test_buffer_add_wraparound(buffer: Buffer) -> None:
	"""Test adding data to the buffer with wraparound."""
	batch_1 = {"a": jnp.array([[1.0], [2.0], [3.0]]), "b": jnp.array([[1, 2], [3, 4], [5, 6]])}
	batch_2 = {"a": jnp.array([[4.0], [5.0], [6.0]]), "b": jnp.array([[7, 8], [9, 10], [11, 12]])}

	buffer = buffer.add(batch_1)
	buffer = buffer.add(batch_2)

	assert buffer.idx == 1
	assert jnp.all(buffer.is_full)
	assert jnp.array_equal(buffer.data["a"], jnp.array([[6.0], [2.0], [3.0], [4.0], [5.0]]))
	assert jnp.array_equal(buffer.data["b"], jnp.array([[11, 12], [3, 4], [5, 6], [7, 8], [9, 10]]))


def test_buffer_sample(buffer: Buffer) -> None:
	"""Test sampling from the buffer."""
	batch = {"a": jnp.array([[1.0], [2.0], [3.0]]), "b": jnp.array([[4, 5], [6, 7], [8, 9]])}

	buffer = buffer.add(batch)

	key = jax.random.key(0)
	sampled_batch = buffer.sample(key, batch_size=2)

	assert isinstance(sampled_batch, dict)
	assert sampled_batch["a"].shape == (2, 1)
	assert sampled_batch["b"].shape == (2, 2)
