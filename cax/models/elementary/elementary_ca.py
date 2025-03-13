"""Elementary Cellular Automata model."""

from cax.core.ca import CA
from flax import nnx

from .elementary_ca_perceive import ElementaryCAPerceive
from .elementary_ca_update import ElementaryCAUpdate


class ElementaryCA(CA):
	"""Elementary Cellular Automata model."""

	def __init__(self, rngs: nnx.Rngs, wolfram_code: str = "01101110"):
		"""Initialize Elementary CA."""
		perceive = ElementaryCAPerceive(rngs=rngs)
		update = ElementaryCAUpdate(wolfram_code)

		super().__init__(perceive, update)
