"""Tests for safe numerics primitives."""

import jax
import jax.numpy as jnp

from cax.utils import safe_divide, safe_norm


def test_safe_divide_values() -> None:
	"""Test that safe_divide matches plain division where valid and is zero elsewhere."""
	numerator = jnp.array([2.0, 4.0, 6.0])
	denominator = jnp.array([1.0, 0.0, 3.0])
	is_valid = denominator > 0.0
	result = safe_divide(numerator, denominator, where=is_valid)
	assert jnp.allclose(result, jnp.array([2.0, 0.0, 2.0]))


def test_safe_divide_gradient_is_finite_at_singularity() -> None:
	"""Test that the gradient is finite where the raw division would produce nan."""

	def loss(denominator: jax.Array) -> jax.Array:
		return jnp.sum(
			safe_divide(jnp.ones_like(denominator), denominator, where=denominator > 0.0)
		)

	grad = jax.grad(loss)(jnp.array([2.0, 0.0]))
	assert jnp.isfinite(grad).all()


def test_safe_norm_values() -> None:
	"""Test that safe_norm matches jnp.linalg.norm on nonzero vectors and is zero at zero."""
	vector = jnp.array([[3.0, 4.0], [0.0, 0.0]])
	result = safe_norm(vector, axis=-1)
	assert jnp.allclose(result, jnp.array([5.0, 0.0]))


def test_safe_norm_gradient_is_finite_at_origin() -> None:
	"""Test that the gradient at the zero vector is finite (zero), unlike a raw norm."""
	grad = jax.grad(lambda vector: jnp.sum(safe_norm(vector, axis=-1)))(jnp.zeros((2, 3)))
	assert jnp.isfinite(grad).all()

	raw_grad = jax.grad(lambda vector: jnp.sum(jnp.linalg.norm(vector, axis=-1)))(jnp.zeros((2, 3)))
	assert jnp.isnan(raw_grad).any()
