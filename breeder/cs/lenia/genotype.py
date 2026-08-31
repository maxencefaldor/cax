"""Lenia genotype module."""

from dataclasses import dataclass

import jax
from jax import Array

from cax.cs.lenia import LeniaRuleParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Genotype:
	"""Lenia genotype: rule parameters and initial state.

	Attributes:
		rule_params: Lenia rule parameters (kernel topology, weights, kernel and growth
			parameters).
		state_init: Initial state with shape `(*spatial_dims, channel_size)`.

	"""

	rule_params: LeniaRuleParams
	state_init: Array
