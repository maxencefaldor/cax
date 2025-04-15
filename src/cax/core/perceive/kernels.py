"""Kernel functions for cellular automata perception."""

import jax.numpy as jnp
from jax import Array


def identity_kernel(ndim: int) -> Array:
	"""Create an identity kernel for the given number of dimensions.

	Args:
		ndim: Number of dimensions for the kernel.

	Returns:
		The identity kernel.

	"""
	kernel = jnp.zeros((3,) * ndim)
	center_index = (1,) * ndim
	kernel = kernel.at[center_index].set(1.0)
	return jnp.expand_dims(kernel, axis=-1)


def neighbors_kernel(ndim: int) -> Array:
	"""Create a neighbors kernel for the given number of dimensions.

	Args:
		ndim: Number of dimensions for the kernel.

	Returns:
		The neighbors kernel.

	"""
	kernel = identity_kernel(ndim)
	return 1.0 - kernel


def grad_kernel(ndim: int, *, normalize: bool = True) -> Array:
	"""Create a gradient kernel for the given number of dimensions.

	Args:
		ndim: Number of dimensions for the kernel.
		normalize: Whether to normalize the kernel.

	Returns:
		The gradient kernel.

	"""
	grad = jnp.array([-1, 0, 1])
	smooth = jnp.array([1, 2, 1])

	kernels = []
	for i in range(ndim):
		kernel = jnp.ones([3] * ndim)

		for j in range(ndim):
			axis_kernel = smooth if i != j else grad
			kernel = kernel * axis_kernel.reshape([-1 if k == j else 1 for k in range(ndim)])

		kernels.append(kernel)

	if normalize:
		kernels = [kernel / jnp.sum(jnp.abs(kernel)) for kernel in kernels]

	return jnp.stack(kernels, axis=-1)


def grad2_kernel(ndim: int, normalize: bool = True) -> Array:
	"""Create a second-order gradient kernel for the given number of dimensions.

	Args:
		ndim: Number of dimensions for the kernel.
		normalize: Whether to normalize the kernel.

	Returns:
		The second-order gradient kernel.

	"""
	kernel = jnp.zeros([3] * ndim)
	center = tuple(1 for _ in range(ndim))
	kernel = kernel.at[center].set(-2.0 * ndim)

	for axis in range(ndim):
		for offset in (-1, 1):
			idx = list(center)
			idx[axis] += offset
			kernel = kernel.at[tuple(idx)].set(1.0)

	if normalize:
		kernel = kernel / jnp.sum(jnp.abs(kernel))

	return jnp.expand_dims(kernel, axis=-1)
