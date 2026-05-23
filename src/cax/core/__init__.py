"""Core abstractions for complex systems.

Exposes the base `ComplexSystem`, `Perceive`, and `Update` interfaces.
"""

from .cs import ComplexSystem
from .perceive import Perceive, Perception
from .update import Update

__all__ = [
	"ComplexSystem",
	"Perceive",
	"Perception",
	"Update",
]
