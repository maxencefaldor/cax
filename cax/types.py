"""Type definitions for CAX."""

from typing import TypeAlias

import jax

State: TypeAlias = jax.Array
Perception: TypeAlias = jax.Array
Input: TypeAlias = jax.Array | None
