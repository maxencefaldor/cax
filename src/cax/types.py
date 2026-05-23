"""Type definitions for CAX.

Provides generic type parameters for the ComplexSystem hierarchy and a
backward-compatible type alias for untyped PyTree contexts.

- PyTree: Any JAX-compatible nested structure (arrays, tuples, lists, dicts, dataclasses, etc.).
- State: TypeVar representing a complex system state. Bound to Array for grid-based CA, or
	custom Pytree dataclasses (e.g. BoidsState) for particle systems.
- Input: TypeVar representing external input to a complex system. Typically None for autonomous
	systems; can be an Array or structured PyTree for conditional or externally-driven systems.

"""

from typing import Any, TypeVar

PyTree = Any

State = TypeVar("State")
Input = TypeVar("Input")
