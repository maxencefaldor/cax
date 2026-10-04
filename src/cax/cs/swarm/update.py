"""Swarm consensus update.

If two of the three neighborhood tags agree on a structured tag (run=2 or
match=3), the cell takes that vote. Otherwise it keeps its own tag. Idle (0)
never wins a vote.
"""

from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.update import Update

RUN = 2.0
MATCH = 3.0


class SwarmUpdate(Update[Array, Array, Array]):
    """Majority vote among left, self, and right for structured tags."""

    @override
    def __call__(
        self, state: Array, perception: Array, input: Array | None = None
    ) -> Array:
        """Vote; structured agreement beats a lone cell.

        Args:
            state: Unused except shape; next tags come from perception.
            perception: ``(..., length, 3)`` left/self/right.
            input: Unused.

        Returns:
            Next state ``(..., length, 1)``.

        """
        left = perception[..., 0]
        mid = perception[..., 1]
        right = perception[..., 2]

        def structured(a: Array) -> Array:
            return jnp.logical_or(jnp.isclose(a, RUN), jnp.isclose(a, MATCH))

        pair_lr = jnp.logical_and(jnp.isclose(left, right), structured(left))
        pair_lm = jnp.logical_and(jnp.isclose(left, mid), structured(left))
        pair_rm = jnp.logical_and(jnp.isclose(right, mid), structured(right))
        nxt = jnp.where(pair_lr, left, mid)
        nxt = jnp.where(pair_lm, left, nxt)
        nxt = jnp.where(pair_rm, right, nxt)
        return nxt[..., None]
