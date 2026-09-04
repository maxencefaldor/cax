"""Tests for the kernel functions."""

import jax.numpy as jnp
import pytest
from jax import Array

from cax.core.perceive import (
    grad2_kernel,
    grad_kernel,
    identity_kernel,
    neighbors_kernel,
)
from cax.core.perceive.kernels import HEX_BASIS, Neighborhood


def test_identity_kernel() -> None:
    """Test the identity_kernel function for 1D, 2D, and 3D cases."""
    # Test 1D identity kernel
    kernel_1d = identity_kernel(num_dims=1)
    expected_1d = jnp.array([[0.0], [1.0], [0.0]])
    assert jnp.allclose(kernel_1d, expected_1d)

    # Test 2D identity kernel
    kernel_2d = identity_kernel(num_dims=2)
    expected_2d = jnp.array(
        [[[0.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]]
    )
    assert jnp.allclose(kernel_2d, expected_2d)

    # Test 3D identity kernel
    kernel_3d = identity_kernel(num_dims=3)
    expected_3d = jnp.zeros((3, 3, 3, 1))
    expected_3d = expected_3d.at[1, 1, 1, 0].set(1.0)
    assert jnp.allclose(kernel_3d, expected_3d)


def test_neighbors_kernel() -> None:
    """Test the neighbors_kernel function for 1D, 2D, and 3D cases."""
    # Test 1D neighbors kernel
    kernel_1d = neighbors_kernel(num_dims=1)
    expected_1d = jnp.array([[[1.0], [0.0], [1.0]]])
    assert jnp.allclose(kernel_1d, expected_1d)

    # Test 2D neighbors kernel
    kernel_2d = neighbors_kernel(num_dims=2)
    expected_2d = jnp.array(
        [[[1.0], [1.0], [1.0]], [[1.0], [0.0], [1.0]], [[1.0], [1.0], [1.0]]]
    )
    assert jnp.allclose(kernel_2d, expected_2d)

    # Test 3D neighbors kernel
    kernel_3d = neighbors_kernel(num_dims=3)
    expected_3d = jnp.ones((3, 3, 3, 1))
    expected_3d = expected_3d.at[1, 1, 1, 0].set(0.0)
    assert jnp.allclose(kernel_3d, expected_3d)


def test_grad_kernel() -> None:
    """Test the grad_kernel function for 1D, 2D, and 3D cases."""
    # Test 1D gradient kernel
    kernel_1d = grad_kernel(num_dims=1)
    expected_1d = jnp.array([[[-0.5], [0.0], [0.5]]])
    assert jnp.allclose(kernel_1d, expected_1d)

    # Test 2D gradient kernel
    kernel_2d = grad_kernel(num_dims=2)
    expected_2d = jnp.array(
        [
            [[-0.125, -0.125], [-0.25, 0.0], [-0.125, 0.125]],
            [[0.0, -0.25], [0.0, 0.0], [0.0, 0.25]],
            [[0.125, -0.125], [0.25, 0.0], [0.125, 0.125]],
        ]
    )
    assert jnp.allclose(kernel_2d, expected_2d)

    # Test 3D gradient kernel
    kernel_3d = grad_kernel(num_dims=3)
    assert kernel_3d.shape == (3, 3, 3, 3)


def test_grad_kernel_not_normalized() -> None:
    """Test the grad_kernel function without normalization."""
    # Test 2D gradient kernel without normalization
    kernel_2d = grad_kernel(num_dims=2, normalize=False)
    expected_2d = jnp.array(
        [
            [[-1.0, -1.0], [-2.0, 0.0], [-1.0, 1.0]],
            [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
            [[1.0, -1.0], [2.0, 0.0], [1.0, 1.0]],
        ]
    )
    assert jnp.allclose(kernel_2d, expected_2d)


@pytest.mark.parametrize("num_dims", [1, 2, 3, 4])
def test_kernel_shapes(num_dims: int) -> None:
    """Test the shapes of kernels for different dimensions."""
    assert identity_kernel(num_dims=num_dims).shape == (3,) * num_dims + (1,)
    assert neighbors_kernel(num_dims=num_dims).shape == (3,) * num_dims + (1,)
    assert grad_kernel(num_dims=num_dims).shape == (3,) * num_dims + (num_dims,)


def test_grad2_kernel_hexagonal() -> None:
    """On a triangular lattice all six neighbors are equidistant, so all weigh alike."""
    kernel = grad2_kernel(num_dims=2, neighborhood="hexagonal", normalize=False)[..., 0]

    # The two corners a triangular lattice does not touch.
    assert kernel[0, 0] == 0.0
    assert kernel[2, 2] == 0.0

    neighbors = kernel[kernel > 0]
    assert neighbors.size == 6
    assert jnp.allclose(neighbors, neighbors[0])
    assert jnp.allclose(jnp.sum(kernel), 0.0)


def test_grad2_kernel_hexagonal_is_isotropic() -> None:
    """The lattice removes the anisotropy the square stencils trade off against.

    Measured as for the square stencils: the response to a quartic, against the same
    quartic rotated. Every direction of the triangular lattice answers alike.
    """
    kernel = grad2_kernel(num_dims=2, neighborhood="hexagonal", normalize=False)[..., 0]
    offsets = jnp.stack(
        jnp.meshgrid(jnp.arange(-1.0, 2.0), jnp.arange(-1.0, 2.0), indexing="ij"),
        axis=-1,
    )
    positions = offsets @ HEX_BASIS

    responses = jnp.array(
        [
            jnp.sum(
                kernel * (positions @ jnp.array([jnp.cos(angle), jnp.sin(angle)])) ** 4
            )
            for angle in jnp.linspace(0.0, jnp.pi, 24)
        ]
    )

    assert float(jnp.max(responses) - jnp.min(responses)) < 1e-5


def test_grad2_kernel_hexagonal_is_two_dimensional() -> None:
    """A triangular lattice is a plane; asking for it in 3D is an error, not a guess."""
    with pytest.raises(ValueError, match="two-dimensional"):
        grad2_kernel(num_dims=3, neighborhood="hexagonal")


def test_grad2_kernel_von_neumann() -> None:
    """The default stencil touches only the face-adjacent cells."""
    kernel = grad2_kernel(num_dims=2, normalize=False)[..., 0]

    assert jnp.array_equal(
        kernel, jnp.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    )


def test_grad2_kernel_moore() -> None:
    """The Moore stencil is the familiar isotropic Laplacian, up to scale."""
    kernel = grad2_kernel(num_dims=2, neighborhood="moore", normalize=False)[..., 0]

    assert jnp.allclose(
        2.0 * kernel, jnp.array([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])
    )


@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("neighborhood", ["von_neumann", "moore"])
def test_grad2_kernel_sums_to_zero(num_dims: int, neighborhood: Neighborhood) -> None:
    """A Laplacian must annihilate a constant field, whatever its support."""
    kernel = grad2_kernel(num_dims=num_dims, neighborhood=neighborhood)

    assert kernel.shape == num_dims * (3,) + (1,)
    assert jnp.allclose(jnp.sum(kernel), 0.0, atol=1e-6)


def test_grad2_kernel_moore_is_less_anisotropic() -> None:
    """The point of the Moore stencil: it depends less on how the grid is oriented.

    Every 3x3 Laplacian is exact on a quadratic, so the anisotropy only shows at fourth
    order. Comparing the response to `x**4` against the same quartic rotated by 45
    degrees, the von Neumann stencil disagrees with itself by half, the Moore one by a
    fifth.
    """
    coordinates = jnp.arange(-1, 2, dtype=jnp.float32)
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    diagonal = (x + y) / jnp.sqrt(2.0)

    def anisotropy(kernel: Array) -> float:
        weights = kernel[..., 0]
        along_axis = jnp.sum(weights * x**4)
        along_diagonal = jnp.sum(weights * diagonal**4)
        return float(
            jnp.abs(along_axis - along_diagonal)
            / jnp.maximum(jnp.abs(along_axis), jnp.abs(along_diagonal))
        )

    assert anisotropy(grad2_kernel(num_dims=2, normalize=False)) == pytest.approx(0.5)
    assert anisotropy(
        grad2_kernel(num_dims=2, neighborhood="moore", normalize=False)
    ) == pytest.approx(0.2)
