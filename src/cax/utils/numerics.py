"""Numerically safe primitives for differentiable simulation.

JAX propagates cotangents through *both* branches of `jnp.where`, so masking an invalid
value after it has been computed leaves `nan` in the gradient even when the forward pass
is finite — the documented "where-NaN" trap. The library-wide convention is therefore to
sanitize the *input* of the unsafe operation, not its output: every division, norm, or
singular kernel evaluated where its argument can be degenerate goes through one of these
helpers (or repeats their double-`where` pattern inline, with a comment naming it).
"""

import jax
import jax.numpy as jnp
from jax import Array


def safe_divide(numerator: Array, denominator: Array, *, where: Array) -> Array:
	"""Divide two arrays, returning zero and a clean gradient where invalid.

	Uses the double-`where` pattern: the denominator is replaced by one at invalid
	positions *before* dividing, so neither the forward value nor the gradient ever
	touches the singular point.

	Args:
		numerator: Numerator array.
		denominator: Denominator array, broadcastable against `numerator`.
		where: Boolean mask, true where the division is valid. Broadcastable against
			the result; invalid positions yield zero.

	Returns:
		`numerator / denominator` where `where` is true, zero elsewhere, with gradients
			that are finite everywhere the inputs are.

	"""
	denominator_safe = jnp.where(where, denominator, jnp.ones_like(denominator))
	return jnp.where(where, numerator / denominator_safe, jnp.zeros_like(numerator))


def safe_norm(vector: Array, *, axis: int = -1, keepdims: bool = False) -> Array:
	"""Euclidean norm with a finite gradient at the origin.

	`jnp.linalg.norm` differentiates to `x / ||x||`, which is `nan` at zero. This
	computes the same value but returns a zero gradient at the origin, which is the
	convention steering and force computations want: a vanished vector exerts no pull.

	Args:
		vector: Input array.
		axis: Axis holding the vector components.
		keepdims: Whether the reduced axis is kept with size one.

	Returns:
		Norm of `vector` along `axis`, with gradient zero where the norm is zero.

	"""
	squared = jnp.sum(jnp.square(vector), axis=axis, keepdims=keepdims)
	is_positive = squared > 0.0
	squared_safe = jnp.where(is_positive, squared, jnp.ones_like(squared))
	return jnp.where(is_positive, jnp.sqrt(squared_safe), jnp.zeros_like(squared))


def detach(tree: Array) -> Array:
	"""Copy a pytree's containers, so storing it cannot alias the caller's object.

	Modules that keep a caller-supplied parameter object are storing part of the
	caller's state. Flax writes a module's state back after a transformed call, so an
	aliased container built inside `jax.jit` or `jax.grad` has tracers written into it
	— corrupting the caller's object and leaking a tracer out of the trace. Rebuilding
	the containers shares the leaf arrays but gives the module its own structure.

	Args:
		tree: A pytree to copy.

	Returns:
		A pytree with the same leaves and freshly built containers.

	"""
	return jax.tree.map(lambda leaf: leaf, tree)
