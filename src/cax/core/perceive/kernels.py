"""Kernel utilities for perception modules.

Each function returns a small spatial kernel suitable for neighborhood aggregation or
finite-difference style operations. Kernels use channel-last layout and a support of
size 3 along each spatial dimension.

`grad2_kernel` takes a `neighborhood`, naming which of the surrounding cells contribute:

- `"von_neumann"` --- the cells sharing a face, two per axis.
- `"moore"` --- every cell in the surrounding block, diagonals included.
- `"hexagonal"` --- a triangular lattice stored in a square array, whose axes are read
  as the basis vectors `(1, 0)` and `(1/2, sqrt(3)/2)` rather than as a Cartesian frame.
  Two opposite corners of the block are then not adjacent, leaving six neighbors that
  are all exactly one unit away. Two dimensions only.
"""

import itertools
import math
from typing import Literal

import jax.numpy as jnp
from jax import Array

Neighborhood = Literal["von_neumann", "moore", "hexagonal"]

HEX_BASIS = jnp.array([[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
"""Rows are the lattice vectors a triangular lattice's array axes stand for."""

_HEX_NON_NEIGHBORS = ((-1, -1), (1, 1))
"""The block corners a triangular lattice does not touch."""


def _square_distance(offset: tuple[int, ...], neighborhood: Neighborhood) -> float:
    """Measure a neighbor's squared distance from the center in the lattice's geometry.

    On a triangular lattice the array axes are not perpendicular, so two cells a
    diagonal step apart can still be nearest neighbors. Measuring through the basis is
    what makes all six of them come out equidistant.
    """
    if neighborhood == "hexagonal":
        position = jnp.asarray(offset, dtype=jnp.float32) @ HEX_BASIS
        return float(jnp.sum(jnp.square(position)))
    return float(sum(step * step for step in offset))


def _neighbor_offsets(
    num_dims: int, neighborhood: Neighborhood
) -> list[tuple[int, ...]]:
    """List the offsets of a neighborhood's neighbors, excluding the center."""
    if neighborhood not in ("von_neumann", "moore", "hexagonal"):
        raise ValueError(f"Unknown neighborhood {neighborhood!r}.")
    if neighborhood == "hexagonal" and num_dims != 2:
        raise ValueError(
            f"The hexagonal neighborhood is two-dimensional, got num_dims={num_dims}."
        )

    offsets = []
    for offset in itertools.product((-1, 0, 1), repeat=num_dims):
        square_distance = sum(step * step for step in offset)
        if not square_distance:
            continue
        if neighborhood == "von_neumann" and square_distance > 1:
            continue
        if neighborhood == "hexagonal" and offset in _HEX_NON_NEIGHBORS:
            continue
        offsets.append(offset)
    return offsets


def identity_kernel(*, num_dims: int) -> Array:
    """Create an identity kernel for the given number of dimensions.

    The kernel has value 1 at the central position and 0 elsewhere.

    Args:
        num_dims: Number of dimensions for the kernel.

    Returns:
        Array with shape `num_dims * (3,) + (1,)`.

    """
    kernel = jnp.zeros(num_dims * (3,))
    center_idx = num_dims * (1,)
    kernel = kernel.at[center_idx].set(1.0)
    return jnp.expand_dims(kernel, axis=-1)


def neighbors_kernel(*, num_dims: int) -> Array:
    """Create a neighbors kernel for the given number of dimensions.

    This kernel is `1 - identity_kernel`, selecting all neighbors and excluding the
    center.

    Args:
        num_dims: Number of dimensions for the kernel.

    Returns:
        Array with shape `num_dims * (3,) + (1,)`.

    """
    kernel = identity_kernel(num_dims=num_dims)
    return 1.0 - kernel


def grad_kernel(*, num_dims: int, normalize: bool = True) -> Array:
    """Create a gradient kernel for the given number of dimensions.

    Args:
        num_dims: Number of dimensions for the kernel.
        normalize: Whether to L1-normalize each axis kernel.

    Returns:
        Array with shape `num_dims * (3,) + (num_dims,)`, one kernel per spatial axis.

    """
    grad = jnp.array([-1, 0, 1])
    smooth = jnp.array([1, 2, 1])

    kernels = []
    for i in range(num_dims):
        kernel = jnp.ones([3] * num_dims)

        for j in range(num_dims):
            axis_kernel = smooth if i != j else grad
            kernel = kernel * axis_kernel.reshape(
                [-1 if k == j else 1 for k in range(num_dims)]
            )

        kernels.append(kernel)

    if normalize:
        kernels = [kernel / jnp.sum(jnp.abs(kernel)) for kernel in kernels]

    return jnp.stack(kernels, axis=-1)


def grad2_kernel(
    *, num_dims: int, neighborhood: Neighborhood = "von_neumann", normalize: bool = True
) -> Array:
    """Create a second-order (Laplacian) kernel.

    Each neighbor is weighted by the inverse square of its distance from the center, and
    the center takes whatever makes the kernel sum to zero, so that it annihilates a
    constant field.

    The neighborhood decides how isotropic the result is. `"von_neumann"` is the
    cheapest stencil, but its leading error favors the grid axes with four-fold symmetry
    --- a bias a system asked to behave the same in every direction will find and use.
    `"moore"` brings in the diagonals, which is the familiar
    `[[1, 2, 1], [2, -12, 2], [1, 2, 1]]` up to the scale `normalize` removes; on a
    quartic rotated by 45 degrees the von Neumann answer moves by 50% and this one by
    20%.

    `"hexagonal"` removes the question. Its six neighbors are all one unit away, so they
    are weighted equally and no direction is singled out at all. Note that an array read
    as a triangular lattice also has to be *drawn* as one --- see
    `cax.utils.hex_to_square`.

    Args:
        num_dims: Number of dimensions for the kernel.
        neighborhood: Which surrounding cells contribute.
        normalize: Whether to L1-normalize the kernel.

    Returns:
        Array with shape `num_dims * (3,) + (1,)`.

    Raises:
        ValueError: If the neighborhood is unknown, or is hexagonal in other than two
            dimensions.

    """
    kernel = jnp.zeros([3] * num_dims)
    for offset in _neighbor_offsets(num_dims, neighborhood):
        kernel = kernel.at[tuple(1 + step for step in offset)].set(
            1.0 / _square_distance(offset, neighborhood)
        )

    center = tuple(1 for _ in range(num_dims))
    kernel = kernel.at[center].set(-jnp.sum(kernel))

    if normalize:
        kernel = kernel / jnp.sum(jnp.abs(kernel))

    return jnp.expand_dims(kernel, axis=-1)
