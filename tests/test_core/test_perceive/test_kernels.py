"""Tests for the kernel functions."""

import jax.numpy as jnp
import pytest
from cax.core.perceive.kernels import grad_kernel, identity_kernel, neighbors_kernel


def test_identity_kernel() -> None:
	"""Test the identity_kernel function for 1D, 2D, and 3D cases."""
	# Test 1D identity kernel
	kernel_1d = identity_kernel(1)
	expected_1d = jnp.array([[0.0], [1.0], [0.0]])
	assert jnp.allclose(kernel_1d, expected_1d)

	# Test 2D identity kernel
	kernel_2d = identity_kernel(2)
	expected_2d = jnp.array([[[0.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]])
	assert jnp.allclose(kernel_2d, expected_2d)

	# Test 3D identity kernel
	kernel_3d = identity_kernel(3)
	expected_3d = jnp.zeros((3, 3, 3, 1))
	expected_3d = expected_3d.at[1, 1, 1, 0].set(1.0)
	assert jnp.allclose(kernel_3d, expected_3d)


def test_neighbors_kernel() -> None:
	"""Test the neighbors_kernel function for 1D, 2D, and 3D cases."""
	# Test 1D neighbors kernel
	kernel_1d = neighbors_kernel(1)
	expected_1d = jnp.array([[[1.0], [0.0], [1.0]]])
	assert jnp.allclose(kernel_1d, expected_1d)

	# Test 2D neighbors kernel
	kernel_2d = neighbors_kernel(2)
	expected_2d = jnp.array([[[1.0], [1.0], [1.0]], [[1.0], [0.0], [1.0]], [[1.0], [1.0], [1.0]]])
	assert jnp.allclose(kernel_2d, expected_2d)

	# Test 3D neighbors kernel
	kernel_3d = neighbors_kernel(3)
	expected_3d = jnp.ones((3, 3, 3, 1))
	expected_3d = expected_3d.at[1, 1, 1, 0].set(0.0)
	assert jnp.allclose(kernel_3d, expected_3d)


def test_grad_kernel() -> None:
	"""Test the grad_kernel function for 1D, 2D, and 3D cases."""
	# Test 1D gradient kernel
	kernel_1d = grad_kernel(1)
	expected_1d = jnp.array([[[-0.5], [0.0], [0.5]]])
	assert jnp.allclose(kernel_1d, expected_1d)

	# Test 2D gradient kernel
	kernel_2d = grad_kernel(2)
	expected_2d = jnp.array(
		[
			[[-0.125, -0.125], [-0.25, 0.0], [-0.125, 0.125]],
			[[0.0, -0.25], [0.0, 0.0], [0.0, 0.25]],
			[[0.125, -0.125], [0.25, 0.0], [0.125, 0.125]],
		]
	)
	assert jnp.allclose(kernel_2d, expected_2d)

	# Test 3D gradient kernel
	kernel_3d = grad_kernel(3)
	assert kernel_3d.shape == (3, 3, 3, 3)


def test_grad_kernel_not_normalized() -> None:
	"""Test the grad_kernel function without normalization."""
	# Test 2D gradient kernel without normalization
	kernel_2d = grad_kernel(2, normalize=False)
	expected_2d = jnp.array(
		[
			[[-1.0, -1.0], [-2.0, 0.0], [-1.0, 1.0]],
			[[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
			[[1.0, -1.0], [2.0, 0.0], [1.0, 1.0]],
		]
	)
	assert jnp.allclose(kernel_2d, expected_2d)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_kernel_shapes(ndim: int) -> None:
	"""Test the shapes of kernels for different dimensions."""
	assert identity_kernel(ndim).shape == (3,) * ndim + (1,)
	assert neighbors_kernel(ndim).shape == (3,) * ndim + (1,)
	assert grad_kernel(ndim).shape == (3,) * ndim + (ndim,)


def test_grad_kernel_normalization() -> None:
	"""Test the normalization of grad_kernel for different dimensions."""
	for ndim in [1, 2, 3]:
		kernel = grad_kernel(ndim)
		for i in range(ndim):
			assert jnp.isclose(jnp.sum(jnp.abs(kernel[..., i])), 1.0)
