"""Flow Lenia update module.

This module implements the update rule for Flow Lenia, which extends Lenia with flow-based
advection. In addition to growth, it computes displacement fields from affinity and
concentration gradients, then applies reintegration tracking to transport matter.

"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.scipy.signal import convolve2d

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import Update

from ..lenia.growth import exponential_growth_fn
from ..lenia.rule import LeniaRuleParams


class FlowLeniaUpdate(Update):
	"""Flow Lenia update rule.

	Extends the standard Lenia update with flow-based advection. Computes affinity fields
	(growth potentials) and their gradients, combines them with concentration gradients to
	produce flow fields, and applies reintegration tracking to transport cell matter through
	space. This creates mass-conservative dynamics with fluid-like behaviors.

	"""

	def __init__(
		self,
		channel_size: int,
		*,
		T: int,
		growth_fn: Callable = exponential_growth_fn,
		rule_params: LeniaRuleParams,
		# Flow Lenia parameters
		theta_A: float = 1.0,
		n: int = 2,
		dd: int = 5,
		sigma: float = 0.65,
	):
		"""Initialize Flow Lenia update.

		Args:
			channel_size: Number of channels.
			T: Time resolution controlling the temporal discretization. Higher values
				produce smoother temporal dynamics with smaller update steps.
			growth_fn: Callable that maps neighborhood potential to growth values. Defines
				how cells respond to their local environment.
			rule_params: Instance of LeniaRuleParams containing kernel and growth parameters
				for each channel.
			theta_A: Threshold value for computing the flow activation alpha. Higher values
				make flow less sensitive to local density.
			n: Exponent controlling the nonlinearity of flow activation. Higher values create
				sharper transitions between flow and no-flow regions.
			dd: Maximum displacement distance in pixels that flow can induce per time step.
				Controls the strength of advective transport.
			sigma: Spread parameter for the displacement kernel. Smaller values create more
				localized flow, larger values produce smoother displacement fields.

		"""
		self.channel_size = channel_size
		self.T = T

		self.weight = rule_params.weight
		self.reshape_kernel_to_channel = self._reshape_kernel_to_channel(rule_params)

		self.growth_fn = growth_fn
		self.growth_params = nnx.data(rule_params.growth_params)

		# Flow Lenia parameters
		self.theta_A = theta_A
		self.n = n
		self.dt = 1 / self.T
		self.dd = dd
		self.sigma = sigma

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		Computes affinity fields from perception, derives flow fields from affinity and
		concentration gradients, and applies reintegration tracking to transport matter
		through space while preserving mass.

		Args:
			state: Array with shape (height, width, channel_size) representing the current state.
			perception: Array with shape (height, width, num_kernels) containing potential fields
				from the perception step.
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (height, width, channel_size) after applying flow-based
				advection and mass redistribution.

		"""
		# Compute growth
		G_k = self.weight * self.growth_fn(
			perception, self.growth_params
		)  # (*spatial_dims, num_rules,)

		# Aggregate growth to channels
		# Affinity map U, previously called growth in Lenia
		U = jnp.dot(G_k, self.reshape_kernel_to_channel)  # (*spatial_dims, channel_size,)

		# Affinity gradient
		nabla_U = sobel(U)  # (*spatial_dims, 2, c)

		# Concentration gradient - diffusion term
		nabla_A = sobel(jnp.sum(state, axis=-1, keepdims=True))  # (*spatial_dims, 2, 1)

		# Weight
		alpha = jnp.clip(
			(state[:, :, None, :] / self.theta_A) ** self.n, 0.0, 1.0
		)  # (*spatial_dims, 1, channel_size)

		# Flow - instantaneous speed of matter
		F = nabla_U * (1 - alpha) - nabla_A * alpha  # (*spatial_dims, 2, channel_size)

		# Reintegration tracking
		state = self.apply_reintegration_tracking(state, F)

		return state

	def apply_reintegration_tracking(self, state: State, F: Array) -> State:
		"""Apply reintegration tracking to transport matter according to flow fields.

		Implements the reintegration tracking algorithm that transports matter from each
		cell to surrounding cells based on the flow field. Matter is distributed according
		to how well each target cell matches the desired displacement, ensuring smooth and
		mass-conservative transport.

		Args:
			state: Array with shape (height, width, channel_size) representing the current
				state before advection.
			F: Flow field with shape (height, width, 2, channel_size) specifying the desired
				displacement for each cell in each channel, where axis -2 is [dy, dx].

		Returns:
			New state with shape (height, width, channel_size) after matter redistribution
				through flow-based advection.

		"""
		SY, SX, C = state.shape

		# Generate all possible displacements
		dys = jnp.arange(-self.dd, self.dd + 1)
		dxs = jnp.arange(-self.dd, self.dd + 1)
		dys, dxs = jnp.meshgrid(dys, dxs, indexing="ij")
		dys = dys.flatten()
		dxs = dxs.flatten()

		# Compute grid positions
		y, x = jnp.arange(SY), jnp.arange(SX)
		Y, X = jnp.meshgrid(y, x, indexing="ij")
		pos = jnp.stack([Y, X], axis=-1) + 0.5  # (SY, SX, 2)

		# Compute target positions (mu)
		ma = self.dd - self.sigma  # Maximum allowed displacement
		F_clipped = jnp.clip(self.dt * F, -ma, ma)  # (SY, SX, 2, C)
		mu = pos[..., None] + F_clipped  # (SY, SX, 2, C)

		# Define step function for each displacement
		def step(dy: Array, dx: Array) -> Array:
			Xr = jnp.roll(state, (dy, dx), axis=(0, 1))  # (SY, SX, C)
			mur = jnp.roll(mu, (dy, dx), axis=(0, 1))  # (SY, SX, 2, C)

			shifts_y = [-SY, 0, SY]
			shifts_x = [-SX, 0, SX]
			dpmu = jnp.min(
				jnp.stack(
					[
						jnp.abs(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None]))
						for di in shifts_y
						for dj in shifts_x
					],
					axis=0,
				),
				axis=0,
			)  # (SY, SX, 2, C)

			sz = 0.5 - dpmu + self.sigma  # (SY, SX, 2, C)
			clipped_sz = jnp.clip(sz, 0, min(1, 2 * self.sigma))  # (SY, SX, 2, C)
			area = jnp.prod(clipped_sz, axis=2) / (4 * self.sigma**2)  # (SY, SX, C)
			nX = Xr * area  # (SY, SX, C)
			return nX

		# Apply step function over all displacements
		nX = jax.vmap(step, in_axes=(0, 0))(dys, dxs)  # (num_displacements, SY, SX, C)
		new_state = jnp.sum(nX, axis=0)  # (SY, SX, C)

		return new_state

	def _reshape_kernel_to_channel(self, rule_params: LeniaRuleParams) -> Array:
		"""Compute array to reshape from kernel to channel."""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size))(
			rule_params.channel_target
		)


def get_sobel_kernels() -> tuple[Array, Array]:
	"""Define Sobel kernels exactly as in the reference Flow Lenia code."""
	kx = jnp.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=jnp.float32)
	ky = jnp.transpose(kx)  # [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
	return kx, ky


def sobel_x(A: Array, kx: Array) -> Array:
	"""Compute horizontal Sobel filter per channel."""
	# Apply convolve2d to each channel using vmap
	return jax.vmap(lambda a: convolve2d(a, kx, mode="same"), in_axes=2, out_axes=2)(A)


def sobel_y(A: Array, ky: Array) -> Array:
	"""Compute vertical Sobel filter per channel."""
	return jax.vmap(lambda a: convolve2d(a, ky, mode="same"), in_axes=2, out_axes=2)(A)


def sobel(A: Array) -> Array:
	"""Compute gradients using Sobel filters, matching the reference Flow Lenia implementation.

	Args:
		A: Input array of shape (y, x, c), where c is the number of channels.

	Returns:
		Gradients of shape (y, x, 2, c), where axis -3 is [sobel_y(A), sobel_x(A)],
		matching the reference code's [-dy, -dx] convention.

	"""
	kx, ky = get_sobel_kernels()

	# Compute gradients per channel
	grad_y = sobel_y(A, ky)  # -dy
	grad_x = sobel_x(A, kx)  # -dx

	# Stack gradients as [grad_y, grad_x] = [-dy, -dx]
	return jnp.stack([grad_y, grad_x], axis=2)  # (y, x, 2, c)
