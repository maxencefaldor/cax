"""Core abstractions for complex systems.

Exposes the base `ComplexSystem` interface and re-exports the `State` and `Input` type
variables.
"""

from cax.types import Input, State

from .cs import ComplexSystem

__all__ = [
	"ComplexSystem",
	"Input",
	"State",
]
