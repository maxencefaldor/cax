"""Type definitions for CAX."""

from typing import Any, TypeAlias

from flax import nnx

PyTree: TypeAlias = Any


class Rule(nnx.Param):
	"""Rule for a CA."""

	pass


State: TypeAlias = PyTree
Perception: TypeAlias = PyTree
Input: TypeAlias = PyTree

Metrics: TypeAlias = PyTree
