"""Update modules for complex systems.

These modules transform a state using a perception and optional input to produce the next
state. Implementations include MLP-based, residual, and Neural Cellular Automata updates.
"""

from .mlp_update import MLPUpdate
from .nca_update import NCAUpdate
from .residual_update import ResidualUpdate
from .update import Update

__all__ = [
	"Update",
	"MLPUpdate",
	"ResidualUpdate",
	"NCAUpdate",
]
