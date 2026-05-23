"""Core abstractions for complex systems.

Exposes the base `ComplexSystem`, `Perceive`, and `Update` interfaces, along with the
backward-compatible `State` and `Input` type aliases.
"""

from cax.types import Input, State

from .cs import ComplexSystem
from .perceive import Perceive
from .update import Update

__all__ = [
	"ComplexSystem",
	"Input",
	"Perceive",
	"State",
	"Update",
]
