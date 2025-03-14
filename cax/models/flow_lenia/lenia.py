"""Lenia model."""

from collections.abc import Callable

from cax.core.ca import CA

from .lenia_perceive import LeniaPerceive, original_kernel_fn
from .lenia_update import LeniaUpdate, growth_exponential
from .types import RuleParams


class Lenia(CA):
	"""Lenia model with optional Flow Lenia support."""

	def __init__(
		self,
		num_dims: int,
		state_size: int,
		R: int,
		T: int,
		rules_params: RuleParams,
		state_scale: float = 1,
		kernel_fn: Callable = original_kernel_fn,
		growth_fn: Callable = growth_exponential,
		flow: bool = False,
		theta_A: float = 1.0,
		n: int = 2,
		dt: float = 0.2,
		dd: int = 5,
		sigma: float = 0.65,
		border: str = "torus",
	):
		"""Initialize Lenia.

		Args:
		    num_dims: Number of spatial dimensions.
		    state_size: Size of each spatial dimension.
		    R: Space resolution.
		    T: Time resolution.
		    rules_params: Parameters for the rules.
		    state_scale: Scaling factor for the state.
		    kernel_fn: Kernel function.
		    growth_fn: Growth function.
		    flow: Whether to use Flow Lenia.
		    theta_A: Threshold for alpha in Flow Lenia.
		    n: Exponent for alpha in Flow Lenia.
		    dt: Time step for flow.
		    dd: Maximum displacement distance.
		    sigma: Spread parameter.
		    border: Boundary condition.

		"""
		perceive = LeniaPerceive(num_dims, state_size, R, state_scale, rules_params, kernel_fn)
		update = LeniaUpdate(
			T, state_scale, rules_params, growth_fn, flow, theta_A, n, dt, dd, sigma, border
		)
		super().__init__(perceive, update)
