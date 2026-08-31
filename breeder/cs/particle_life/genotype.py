"""Particle Life genotype module."""

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Genotype:
	"""Particle Life genotype: the interaction rule and the initial arrangement.

	Attributes:
		attraction: Attraction matrix with shape `(num_classes, num_classes)`, entries in
			[-1, 1]; entry `(i, j)` is the pull of class `j` on class `i` (negative
			repels). Asymmetry is the point — it is what makes chase, orbit and
			self-organizing structure possible.
		beta: Crossover radius of the force law, as a fraction of the interaction radius:
			particles closer than `beta · r_max` always repel, farther ones follow the
			attraction matrix. The one dimensionless shape parameter of the force.
		position_init: Initial positions with shape `(num_particles, 2)` in [0, 1).

	"""

	attraction: Array
	beta: Array
	position_init: Array
