"""Elementary Cellular Automata model."""

from collections.abc import Callable

from cax.core.ca import CA, metrics_fn
from flax import nnx

from .elementary_ca_perceive import ElementaryCAPerceive
from .elementary_ca_update import ElementaryCAUpdate


class ElementaryCA(CA):
	"""Elementary Cellular Automata model."""

	def __init__(self, rngs: nnx.Rngs, *, wolfram_code: str = "01101110", metrics_fn: Callable = metrics_fn):
		"""Initialize Elementary CA."""
		perceive = ElementaryCAPerceive(rngs=rngs)
		update = ElementaryCAUpdate(wolfram_code=wolfram_code)
		super().__init__(perceive, update, metrics_fn=metrics_fn)
