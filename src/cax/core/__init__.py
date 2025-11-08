"""Core abstractions for complex systems.

Exposes the base `ComplexSystem` interface and re-exports the `State` and `Input` type aliases.
"""

from .cs import ComplexSystem, Input, State

__all__ = [
	"ComplexSystem",
	"Input",
	"State",
]
