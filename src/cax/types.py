"""Type aliases.

These aliases centralize common structural types. They intentionally remain broad to support
JAX/Flax transformations (jit, vmap, scan) over arbitrarily nested container structures.

- PyTree: Any JAX-compatible nested structure (arrays, tuples, lists, dicts, dataclasses, etc.).
- State: Alias for ``PyTree`` representing a complex system state; shapes and dtypes are
	context-dependent and documented at call sites.
- Input: Alias for ``PyTree`` representing optional external inputs to updates; when unused,
	``None`` is typically passed.

"""

from typing import Any

PyTree = Any

State = PyTree
Input = PyTree
