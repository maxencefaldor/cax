"""Type aliases for CAX.

- PyTree: Any JAX-compatible nested structure (arrays, tuples, lists, dicts, dataclasses, etc.).
- Perception: Output type of a perceive module; commonly an array or tuple of arrays.
- State: Backward-compatible alias for use in user code and examples.
- Input: Backward-compatible alias for use in user code and examples.

"""

from typing import Any

PyTree = Any

Perception = PyTree

State = Any
Input = Any
