"""Flow Lenia module.

This module implements Flow Lenia, a mass-conservative extension of Lenia.
"""

from collections.abc import Callable

from ..lenia.cs import Lenia
from ..lenia.growth import exponential_growth_fn
from ..lenia.kernel import gaussian_kernel_fn
from ..lenia.perceive import LeniaPerceive
from ..lenia.rule import LeniaRuleParams
from .update import FlowLeniaUpdate


class FlowLenia(Lenia):
	"""Flow Lenia class.

	Subclasses Lenia: perception and rendering are Lenia's, while the update adds
	flow-based advection with mass conservation. Like Lenia, Flow Lenia works in any
	number of spatial dimensions.
	"""

	def __init__(
		self,
		*,
		spatial_dims: tuple[int, ...],
		channel_size: int,
		R: int,
		T: float,
		state_scale: float = 1.0,
		kernel_fn: Callable = gaussian_kernel_fn,
		growth_fn: Callable = exponential_growth_fn,
		rule_params: LeniaRuleParams,
		# Flow Lenia parameters
		theta_A: float | None = None,
		n: int = 2,
		dd: int = 5,
		sigma: float = 0.65,
	):
		"""Initialize Flow Lenia.

		Args:
			spatial_dims: Spatial dimensions (e.g., (64, 64) for 2D or (32, 32, 32) for 3D).
				Note that reintegration tracking considers (2 * dd + 1) ** num_spatial_dims
				displacements per cell, which sets the memory and compute cost.
			channel_size: Number of channels.
			R: Space resolution defining the kernel radius. Larger values create wider
				neighborhoods and smoother patterns.
			T: Time resolution controlling the temporal discretization. Higher values
				produce smoother temporal dynamics with smaller update steps.
			state_scale: Scaling factor applied to state values.
			kernel_fn: Callable that generates convolution kernels. Takes rule parameters
				and returns kernel weights.
			growth_fn: Callable that maps neighborhood potential to growth values. Defines
				how cells respond to their local environment.
			rule_params: Instance of LeniaRuleParams containing kernel and growth parameters
				for each channel.
			theta_A: Threshold value for computing the flow activation alpha. Higher values
				make flow less sensitive to local density. Defaults to ``channel_size``,
				matching the official implementation.
			n: Exponent controlling the nonlinearity of flow activation. Higher values create
				sharper transitions between flow and no-flow regions.
			dd: Maximum displacement distance in pixels that flow can induce per time step.
				Controls the strength of advective transport.
			sigma: Spread parameter for the displacement kernel. Smaller values create more
				localized flow, larger values produce smoother displacement fields.

		"""
		self.perceive = LeniaPerceive(
			spatial_dims=spatial_dims,
			channel_size=channel_size,
			R=R,
			state_scale=state_scale,
			kernel_fn=kernel_fn,
			rule_params=rule_params,
		)
		self.update = FlowLeniaUpdate(
			channel_size=channel_size,
			T=T,
			growth_fn=growth_fn,
			rule_params=rule_params,
			theta_A=theta_A,
			n=n,
			dd=dd,
			sigma=sigma,
		)
