"""Life model."""

from cax.core.ca import CA
from flax import nnx

from .life_perceive import LifePerceive
from .life_update import LifeUpdate


class Life(CA):
	"""Life model."""

	def __init__(self, rngs: nnx.Rngs):
		"""Initialize Life."""
		perceive = LifePerceive(rngs=rngs)
		update = LifeUpdate()

		super().__init__(perceive, update)
