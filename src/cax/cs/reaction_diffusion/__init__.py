"""Reaction-Diffusion (Gray-Scott) module.

References:
	[1] Autocatalytic reactions in the isothermal, continuous stirred tank reactor,
		Gray and Scott. 1984.
	[2] Complex Patterns in a Simple System, John E. Pearson. 1993.

"""

from .cs import ReactionDiffusion
from .perceive import ReactionDiffusionPerceive
from .update import ReactionDiffusionUpdate

__all__ = [
	"ReactionDiffusion",
	"ReactionDiffusionPerceive",
	"ReactionDiffusionUpdate",
]
