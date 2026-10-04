"""Swarm 1-D perception: left, self, right.

Isolated workers zero the neighbor that sits across a chunk boundary so
decode of each worker never reads another worker. Chained mode keeps the
neighbor so history can cross the cut.
"""

from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.perceive import Perceive


class SwarmPerceive(Perceive[Array, Array]):
    """Left/self/right tags along a 1-D line of workers."""

    def __init__(self, *, worker_size: int, isolated: bool):
        """Initialize swarm perceive.

        Args:
            worker_size: Cells per isolated worker.
            isolated: If True, neighbors across a worker boundary are idle (0).

        """
        self.worker_size = worker_size
        self.isolated = isolated

    @override
    def __call__(self, state: Array) -> Array:
        """Return channels ``(left, self, right)``.

        Args:
            state: Array ``(..., length, 1)`` of tag ids.

        Returns:
            Array ``(..., length, 3)``.

        """
        tags = state[..., 0]
        left = jnp.roll(tags, 1, axis=-1)
        right = jnp.roll(tags, -1, axis=-1)
        if self.isolated:
            length = tags.shape[-1]
            idx = jnp.arange(length)
            at_start = (idx % self.worker_size) == 0
            at_end = (idx % self.worker_size) == (self.worker_size - 1)
            left = jnp.where(at_start, 0.0, left)
            right = jnp.where(at_end, 0.0, right)
        return jnp.stack([left, tags, right], axis=-1)
