"""Type definitions for CAX."""

import jax
from typing import TypeAlias

State: TypeAlias = jax.Array
Perception: TypeAlias = jax.Array
Input: TypeAlias = jax.Array | None
