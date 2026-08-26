"""Elementary Cellular Automata module.

References:
	[1] A New Kind of Science, Stephen Wolfram. 2002.

"""

from .cs import Elementary
from .perceive import ElementaryPerceive
from .update import ElementaryUpdate

__all__ = [
	"Elementary",
	"ElementaryPerceive",
	"ElementaryUpdate",
]
