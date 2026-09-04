"""Lenia update module.

This module implements the update rule for Lenia, which applies a growth mapping to the
potential fields and updates the state. Growth values are computed per-kernel using a
parameterized growth function, then aggregated back to channels.
"""

from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.update import Update

from .growth import LeniaGrowthParams, exponential_growth_fn
from .rule import LeniaRuleParams


class LeniaUpdate(Update[Array, Array, Array]):
    """Lenia update rule.

    Applies the growth mapping to potential fields to determine how much each cell
    should grow or decay. Growth is computed per-kernel using parameterized growth
    functions, weighted, and aggregated to target channels. The state is updated in
    discrete time steps with temporal resolution T.
    """

    def __init__(
        self,
        *,
        channel_size: int,
        T: float,
        growth_fn: Callable[[Array, LeniaGrowthParams], Array] = exponential_growth_fn,
        rule_params: LeniaRuleParams,
    ):
        """Initialize Lenia update.

        Args:
            channel_size: Number of channels.
            T: Time resolution controlling the temporal discretization. Higher values
                produce smoother temporal dynamics with smaller update steps.
            growth_fn: Callable that maps neighborhood potential to growth values.
                Defines how cells respond to their local environment.
            rule_params: Instance of LeniaRuleParams containing kernel and growth
                parameters for each channel.

        """
        self.channel_size = channel_size
        self.T = T

        # Chan's multi-kernel convention ("Lenia and Expanded Universe", 2020): growth
        # is the weighted AVERAGE of per-kernel growths, factors h_k / sum(h). Flow
        # Lenia inherits this through _growth, where the reference uses raw h instead —
        # see FlowLeniaUpdate's fidelity notes.
        self.normalized_weight = rule_params.weight / jnp.sum(rule_params.weight)
        self.reshape_kernel_to_channel = self._reshape_kernel_to_channel(rule_params)

        self.growth_fn = growth_fn
        self.growth_params = nnx.data(rule_params.growth_params)

    @override
    def __call__(
        self, state: Array, perception: Array, input: Array | None = None
    ) -> Array:
        """Process the current state, perception, and input to produce a new state.

        Computes growth values from potential fields using the growth function,
        aggregates them to target channels, and updates the state with temporal
        resolution T. The updated state is clipped to [0, 1].

        Args:
            state: Array with shape (*spatial_dims, channel_size) representing the
                current state.
            perception: Array with shape (*spatial_dims, num_kernels) containing
                potential fields from the perception step.
            input: Optional input (unused in this implementation).

        Returns:
            Next state with shape (*spatial_dims, channel_size) after applying growth
            and clipping to [0, 1].

        """
        # Update state with the aggregated growth and clip
        state = jnp.clip(state + self._growth(perception) / self.T, 0.0, 1.0)

        return state

    def _growth(self, perception: Array) -> Array:
        """Compute per-kernel growth and aggregate it to channels.

        Args:
            perception: Array with shape (*spatial_dims, num_kernels) of potential
                fields.

        Returns:
            Array with shape (*spatial_dims, channel_size) of aggregated growth.

        """
        G_k = self.normalized_weight * self.growth_fn(perception, self.growth_params)
        return jnp.dot(G_k, self.reshape_kernel_to_channel)

    def _reshape_kernel_to_channel(self, rule_params: LeniaRuleParams) -> Array:
        """Compute array to reshape from kernel to channel.

        Returns a matrix `K -> C` that aggregates per-kernel growth into channel space
        using `rule_params.channel_target`.

        Args:
            rule_params: Rule parameters containing the `channel_target` mapping.

        Returns:
            Array with shape `(K, C)` suitable for `jnp.dot(G_k, reshape)`.

        """
        return jax.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size))(
            rule_params.channel_target
        )
