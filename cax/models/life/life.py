"""Life model."""

from collections.abc import Callable

from cax.core.ca import CA, metrics_fn
from flax import nnx

from .life_perceive import LifePerceive
from .life_update import LifeUpdate


class Life(CA):
	"""Life model."""

	def __init__(self, rngs: nnx.Rngs, *, metrics_fn: Callable = metrics_fn):
		"""Initialize Life."""
		perceive = LifePerceive(rngs=rngs)
		update = LifeUpdate()
		super().__init__(perceive, update, metrics_fn=metrics_fn)
