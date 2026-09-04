"""Flow Lenia update module.

This module implements the update rule for Flow Lenia, which extends Lenia with
flow-based advection. In addition to growth, it computes displacement fields from
affinity and concentration gradients, then applies reintegration tracking to transport
matter.

References:
    [1] Flow-Lenia: Towards open-ended evolution in cellular automata through mass
        conservation and parameter localization, Plantec et al. 2023. arXiv:2212.07906.
        Official implementation: https://github.com/erwanplantec/FlowLenia (flowlenia.py
        is the variant mirrored here; verified bitwise-identical on shared configs
        away from the grid boundary — see the boundary note below).

"""

from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..lenia.growth import LeniaGrowthParams, exponential_growth_fn
from ..lenia.rule import LeniaRuleParams
from ..lenia.update import LeniaUpdate


class FlowLeniaUpdate(LeniaUpdate):
    """Flow Lenia update rule.

    Extends the standard Lenia update with flow-based advection. Computes affinity
    fields (growth potentials) and their gradients, combines them with concentration
    gradients to produce flow fields, and applies reintegration tracking to transport
    cell matter through space. This creates mass-conservative dynamics with fluid-like
    behaviors.

    Reference fidelity, stated where it can be checked:

    - The flow gate alpha is computed per channel from A, matching the official
        implementation ([1], flowlenia.py); the paper's Eq. 5 gates by the channel-sum
        density instead. The two coincide for channel_size 1.
    - The official implementation hardcodes theta_A = channel_size and n = 2; both are
        parameters here, with defaults reproducing the official behavior.
    - The Sobel gradients wrap the torus, consistent with the perception and the
        reintegration; the official implementation zero-pads them at the boundary,
        contradicting its own periodic domain. Away from the boundary the two are
        bitwise identical.
    - Kernel weights are normalized to sum to one (Chan's Lenia convention, inherited
        from LeniaUpdate's _growth); the reference uses raw h. Identical when sum(h) is
        one. In Flow Lenia the scale of h is a real degree of freedom — it sets the
        balance of affinity flow against the alpha-gated diffusion term — and can be
        recovered by scaling the growth function.

    """

    def __init__(
        self,
        *,
        channel_size: int,
        T: float,
        growth_fn: Callable[[Array, LeniaGrowthParams], Array] = exponential_growth_fn,
        rule_params: LeniaRuleParams,
        # Flow Lenia parameters
        theta_A: float | None = None,
        n: int = 2,
        dd: int = 5,
        sigma: float = 0.65,
    ):
        """Initialize Flow Lenia update.

        Args:
            channel_size: Number of channels.
            T: Time resolution controlling the temporal discretization. Higher values
                produce smoother temporal dynamics with smaller update steps.
            growth_fn: Callable that maps neighborhood potential to growth values.
                Defines how cells respond to their local environment.
            rule_params: Instance of LeniaRuleParams containing kernel and growth
                parameters for each channel.
            theta_A: Threshold value for computing the flow activation alpha. Higher
                values make flow less sensitive to local density. Defaults to
                ``channel_size``, matching the official implementation.
            n: Exponent controlling the nonlinearity of flow activation. Higher values
                create sharper transitions between flow and no-flow regions.
            dd: Maximum displacement distance in pixels that flow can induce per time
                step. Controls the strength of advective transport.
            sigma: Spread parameter for the displacement kernel. Smaller values create
                more localized flow, larger values produce smoother displacement fields.

        """
        super().__init__(
            channel_size=channel_size, T=T, growth_fn=growth_fn, rule_params=rule_params
        )

        # Flow Lenia parameters
        self.theta_A = channel_size if theta_A is None else theta_A
        self.n = n
        self.dd = dd
        self.sigma = sigma

    @override
    def __call__(
        self, state: Array, perception: Array, input: Array | None = None
    ) -> Array:
        """Process the current state, perception, and input to produce a new state.

        Computes affinity fields from perception, derives flow fields from affinity and
        concentration gradients, and applies reintegration tracking to transport matter
        through space while preserving mass.

        Args:
            state: Array with shape (*spatial_dims, channel_size) representing the
                current state.
            perception: Array with shape (*spatial_dims, num_kernels) containing
                potential fields from the perception step.
            input: Optional input (unused in this implementation).

        Returns:
            Next state with shape (*spatial_dims, channel_size) after applying
            flow-based advection and mass redistribution.

        """
        # Affinity map U: Lenia's aggregated growth, reused from the parent update
        U = self._growth(perception)  # (*spatial_dims, channel_size)

        # Affinity gradient
        nabla_U = sobel(U)  # (*spatial_dims, num_spatial_dims, c)

        # Concentration gradient - diffusion term
        nabla_A = sobel(
            jnp.sum(state, axis=-1, keepdims=True)
        )  # (*spatial_dims, num_spatial_dims, 1)

        # Weight
        alpha = jnp.clip(
            (state[..., None, :] / self.theta_A) ** self.n, 0.0, 1.0
        )  # (*spatial_dims, 1, channel_size)

        # Flow - instantaneous speed of matter
        F = (
            nabla_U * (1 - alpha) - nabla_A * alpha
        )  # (*spatial_dims, num_spatial_dims, channel_size)

        # Reintegration tracking
        state = self.apply_reintegration_tracking(state, F)

        return state

    def apply_reintegration_tracking(self, state: Array, F: Array) -> Array:
        """Apply reintegration tracking to transport matter according to flow fields.

        Implements the reintegration tracking algorithm that transports matter from each
        cell to surrounding cells based on the flow field. Matter is distributed
        according to how well each target cell matches the desired displacement,
        ensuring smooth and mass-conservative transport. The number of candidate
        displacements grows as ``(2 * dd + 1) ** num_spatial_dims``, which sets the
        memory and compute cost.

        Args:
            state: Array with shape (*spatial_dims, channel_size) representing the
                current state before advection.
            F: Flow field with shape (*spatial_dims, num_spatial_dims, channel_size)
                specifying the desired displacement for each cell in each channel, where
                axis -2 orders the components by spatial axis.

        Returns:
            New state with shape (*spatial_dims, channel_size) after matter
            redistribution through flow-based advection.

        """
        *spatial_dims, _channel_size = state.shape
        num_spatial_dims = len(spatial_dims)
        spatial_axes = tuple(range(num_spatial_dims))
        dt = (
            1 / self.T
        )  # The reference's symbol; T is the stored convention family-wide

        # Generate all possible displacements: (num_displacements, num_spatial_dims)
        steps = [jnp.arange(-self.dd, self.dd + 1)] * num_spatial_dims
        displacements = jnp.stack(jnp.meshgrid(*steps, indexing="ij"), axis=-1)
        displacements = displacements.reshape(-1, num_spatial_dims)

        # Compute grid positions, shape (*spatial_dims, num_spatial_dims)
        coordinates = [jnp.arange(dim) for dim in spatial_dims]
        pos = jnp.stack(jnp.meshgrid(*coordinates, indexing="ij"), axis=-1) + 0.5

        # Compute target positions (mu)
        ma = self.dd - self.sigma  # Maximum allowed displacement
        F_clipped = jnp.clip(dt * F, -ma, ma)  # (*spatial_dims, num_spatial_dims, C)
        mu = pos[..., None] + F_clipped  # (*spatial_dims, num_spatial_dims, C)

        # Torus images: every combination of shifting each axis by -size, 0, or +size
        images = [jnp.array([-dim, 0, dim]) for dim in spatial_dims]
        shifts = jnp.stack(jnp.meshgrid(*images, indexing="ij"), axis=-1)
        shifts = shifts.reshape(-1, num_spatial_dims)

        # Define step function for each displacement
        def step(displacement: Array) -> Array:
            Xr = jnp.roll(state, displacement, axis=spatial_axes)
            mur = jnp.roll(mu, displacement, axis=spatial_axes)

            dpmu = jnp.min(
                jnp.stack(
                    [
                        jnp.abs(pos[..., None] - (mur + shift[..., :, None]))
                        for shift in shifts
                    ],
                    axis=0,
                ),
                axis=0,
            )  # (*spatial_dims, num_spatial_dims, C)

            sz = 0.5 - dpmu + self.sigma  # (*spatial_dims, num_spatial_dims, C)
            clipped_sz = jnp.clip(sz, 0, min(1, 2 * self.sigma))
            area = jnp.prod(clipped_sz, axis=-2) / (2 * self.sigma) ** num_spatial_dims
            nX = Xr * area  # (*spatial_dims, C)
            return nX

        # Apply step function over all displacements
        nX = jax.vmap(step)(displacements)  # (num_displacements, *spatial_dims, C)
        new_state = jnp.sum(nX, axis=0)  # (*spatial_dims, C)

        return new_state


def get_sobel_kernels(num_spatial_dims: int) -> Array:
    """Build the n-dimensional Sobel kernels, one per spatial axis.

    Each kernel is the outer product of the derivative stencil [1, 0, -1] along its
    axis and the smoothing stencil [1, 2, 1] along every other axis. For two
    dimensions this reproduces the reference Flow Lenia Sobel filters exactly.

    Args:
        num_spatial_dims: Number of spatial dimensions.

    Returns:
        Array of shape (num_spatial_dims, *(3,) * num_spatial_dims) stacking the kernel
            for each spatial axis.

    """
    smooth = jnp.array([1.0, 2.0, 1.0], dtype=jnp.float32)
    derivative = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    kernels = []
    for axis in range(num_spatial_dims):
        kernel = jnp.array(1.0, dtype=jnp.float32)
        for d in range(num_spatial_dims):
            kernel = kernel[..., None] * (derivative if d == axis else smooth)
        kernels.append(kernel)
    return jnp.stack(kernels)


def sobel(A: Array) -> Array:
    """Compute gradients with Sobel filters, matching the reference Flow Lenia code.

    The domain is a torus, so the input is wrap-padded before the stencil is applied:
    differentiation then commutes with translation exactly. This is a stated deviation
    from the official implementation, whose `convolve2d(mode="same")` zero-pads the
    boundary while its perception and reintegration both wrap — away from the boundary
    the two agree to float32 roundoff.

    All axes and channels are computed by a single grouped convolution. In isolation
    that is several times faster than one convolution per axis; inside the fused
    update step the two measure the same on CPU, and the grouped form is kept for the
    single-kernel structure it hands to other backends. The stencil is fixed physics,
    so it stays a plain constant rather than an `nnx.Conv`'s trainable weights. The
    grouped accumulation orders the float32 sums differently from the reference's
    `convolve2d`, a deviation of at most one unit in the last place — the same scale
    at which any fixed summation order breaks exact symmetry under axis permutation.

    Args:
        A: Input array of shape (*spatial_dims, c), where c is the number of channels.

    Returns:
        Gradients of shape (*spatial_dims, num_spatial_dims, c), where axis -2 orders
        the components by spatial axis. Each component is +2 * 4^(num_spatial_dims - 1)
        times the partial derivative along its axis — in two dimensions, the reference
        implementation's [sobel_y(A), sobel_x(A)].

    """
    num_spatial_dims = A.ndim - 1
    channel_size = A.shape[-1]
    kernels = get_sobel_kernels(num_spatial_dims)

    # The reference convolves (flipping the kernel); lax.conv cross-correlates, so flip.
    flipped = jnp.flip(kernels, axis=tuple(range(1, num_spatial_dims + 1)))

    # One grouped convolution, one group per channel, producing every axis gradient
    # for that channel; rhs shape (*window, 1, channel_size * num_spatial_dims).
    rhs = jnp.tile(
        jnp.moveaxis(flipped, 0, -1)[..., None, :],
        (*([1] * num_spatial_dims), 1, channel_size),
    )
    rhs = rhs.reshape(*([3] * num_spatial_dims), 1, channel_size * num_spatial_dims)

    # Wrap-pad the spatial axes so every stencil window lives on the torus
    padded = jnp.pad(A, [(1, 1)] * num_spatial_dims + [(0, 0)], mode="wrap")

    # Channel-last layouts for any dimensionality: lhs (N, *spatial, C),
    # rhs (*window, I, O), out (N, *spatial, C).
    dimension_numbers = jax.lax.ConvDimensionNumbers(
        lhs_spec=(0, num_spatial_dims + 1, *range(1, num_spatial_dims + 1)),
        rhs_spec=(num_spatial_dims + 1, num_spatial_dims, *range(num_spatial_dims)),
        out_spec=(0, num_spatial_dims + 1, *range(1, num_spatial_dims + 1)),
    )
    gradients = jax.lax.conv_general_dilated(
        padded[None],
        rhs,
        window_strides=(1,) * num_spatial_dims,
        padding="VALID",
        dimension_numbers=dimension_numbers,
        feature_group_count=channel_size,
    )[0]  # (*spatial_dims, channel_size * num_spatial_dims)

    gradients = gradients.reshape(*A.shape[:-1], channel_size, num_spatial_dims)
    return jnp.moveaxis(gradients, -1, -2)  # (*spatial_dims, num_spatial_dims, c)
