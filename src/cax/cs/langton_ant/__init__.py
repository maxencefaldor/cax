"""Langton's Ant module.

References:
	[1] Studying artificial life with cellular automata, Christopher G. Langton. 1986.

"""

from .cs import LangtonAnt
from .perceive import LangtonAntPerceive
from .state import LangtonAntState
from .update import LangtonAntUpdate

__all__ = [
	"LangtonAnt",
	"LangtonAntPerceive",
	"LangtonAntState",
	"LangtonAntUpdate",
]
