"""Tests for the CA class."""

import jax.numpy as jnp
import pytest
from cax.core.ca import CA
from cax.core.perceive.perceive import Perceive
from cax.core.update.update import Update
from flax import nnx


class DummyPerceive(Perceive):
	"""A dummy perceive class for testing."""

	def __call__(self, state):
		"""Multiply the state by 2."""
		return state * 2


class DummyUpdate(Update):
	"""A dummy update class for testing."""

	def __call__(self, state, perception, input=None):
		"""Add 1 to the perception."""
		return perception + 1


@pytest.fixture
def ca():
	"""Create a CA instance for testing."""
	perceive = DummyPerceive()
	update = DummyUpdate()
	return CA(perceive, update)


def test_ca_initialization():
	"""Test the initialization of the CA class."""
	perceive = DummyPerceive()
	update = DummyUpdate()
	ca = CA(perceive, update)
	assert isinstance(ca.perceive, DummyPerceive)
	assert isinstance(ca.update, DummyUpdate)


def test_ca_step(ca):
	"""Test the step method of the CA class."""
	state = jnp.array([1.0, 2.0, 3.0])
	new_state = ca.step(state)
	expected = jnp.array([3.0, 5.0, 7.0])
	assert jnp.allclose(new_state, expected)


def test_ca_call_single_step(ca):
	"""Test the CA class with a single step."""
	state = jnp.array([1.0, 2.0, 3.0])
	new_state = ca(state, num_steps=1)
	expected = jnp.array([3.0, 5.0, 7.0])
	assert jnp.allclose(new_state, expected)


def test_ca_call_multiple_steps(ca):
	"""Test the CA class with multiple steps."""
	state = jnp.array([1.0, 2.0, 3.0])
	new_state = ca(state, num_steps=3)
	expected = jnp.array([15.0, 23.0, 31.0])
	assert jnp.allclose(new_state, expected)


def test_ca_call_all_steps(ca):
	"""Test the CA class with all steps returned."""
	state = jnp.array([1.0, 2.0, 3.0])
	states = ca(state, num_steps=3, all_steps=True)
	expected = jnp.array([[3.0, 5.0, 7.0], [7.0, 11.0, 15.0], [15.0, 23.0, 31.0]])
	assert jnp.allclose(states, expected)


def test_ca_call_with_input():
	"""Test the CA class with input."""

	class InputUpdate(Update):
		"""An update class that uses input."""

		def __call__(self, state, perception, input=None):
			"""Add input to perception if provided."""
			return perception + input if input is not None else perception

	perceive = DummyPerceive()
	update = InputUpdate()
	ca = CA(perceive, update)

	state = jnp.array([1.0, 2.0, 3.0])
	input = jnp.array([0.1, 0.2, 0.3])
	new_state = ca(state, input, num_steps=1)
	expected = jnp.array([2.1, 4.2, 6.3])
	assert jnp.allclose(new_state, expected)


def test_ca_call_with_input_multiple_steps():
	"""Test the CA class with input over multiple steps."""

	class InputUpdate(Update):
		"""An update class that uses input."""

		def __call__(self, state, perception, input=None):
			"""Add input to perception if provided."""
			return perception + input if input is not None else perception

	perceive = DummyPerceive()
	update = InputUpdate()
	ca = CA(perceive, update)

	state = jnp.array([1.0, 2.0, 3.0])
	input = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
	new_state = ca(state, input, num_steps=2, input_in_axis=0)
	expected = jnp.array([19.0, 26.0, 33.0])
	assert jnp.allclose(new_state, expected)


def test_ca_vmap(ca):
	"""Test vectorized mapping of the CA class."""
	states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	new_states = nnx.vmap(lambda state: ca(state, num_steps=3))(states)
	expected = jnp.array([[15.0, 23.0, 31.0], [39.0, 47.0, 55.0]])
	assert jnp.allclose(new_states, expected)
