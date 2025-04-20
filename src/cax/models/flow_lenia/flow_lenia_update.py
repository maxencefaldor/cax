"""Flow Lenia update module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.scipy.signal import convolve2d

from cax.core.update.update import Update
from cax.types import Input, Perception, State

from ..lenia.growth import exponential_growth_fn
from ..lenia.rule import RuleParams


class FlowLeniaUpdate(Update):
	"""Flow Lenia update class."""

	def __init__(
		self,
		channel_size: int,
		T: int,
		rule_params: RuleParams,
		*,
		growth_fn: Callable = exponential_growth_fn,
		# Flow Lenia parameters
		theta_A: float = 1.0,
		n: int = 2,
		dd: int = 5,
		sigma: float = 0.65,
	):
		"""Initialize the FlowLeniaUpdate.

		Args:
			channel_size: Number of channels.
			T: Time resolution.
			rule_params: Parameters for the rules.
			growth_fn: Growth function.
			theta_A: Threshold for alpha computation in Flow Lenia.
			n: Exponent for alpha computation in Flow Lenia.
			dt: Time step for flow displacement.
			dd: Maximum displacement distance.
			sigma: Spread parameter for reintegration tracking.

		"""
		super().__init__()
		self.channel_size = channel_size
		self.T = T
		self.growth_fn = growth_fn

		# Flow Lenia parameters
		self.theta_A = theta_A
		self.n = n
		self.dt = 1 / self.T
		self.dd = dd
		self.sigma = sigma

		# Set rule parameters
		self.rule_params = jax.tree.map(nnx.Param, rule_params)

		# Reshape kernel to channel
		self.reshape_kernel_to_channel = nnx.Param(
			jax.vmap(lambda x: jax.nn.one_hot(x, num_classes=channel_size))(
				rule_params.channel_target
			)
		)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Flow Lenia update.

		Args:
			state: Current state of shape (y, x, c).
			perception: Perceived state of shape (y, x, k).
			input: External input (unused).

		Returns:
			Updated state of shape (y, x, c).

		"""
		# Compute growth
		G_k = self.rule_params.weight * self.growth_fn(
			perception, self.rule_params.growth_params
		)  # (*spatial_dims, num_rules,)

		# Aggregate growth to channels
		# Affinity map U, previously called growth in Lenia
		U = jnp.dot(G_k, self.reshape_kernel_to_channel.value)  # (*spatial_dims, channel_size,)

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
		"""Apply reintegration tracking to update the state based on the flow field.

		Args:
			state: Current state of shape (y, x, c).
			F: Flow field of shape (y, x, 2, c).

		Returns:
			New state of shape (y, x, c).

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
		def step(dy, dx):
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


def get_sobel_kernels():
	"""Define Sobel kernels exactly as in the reference Flow Lenia code."""
	kx = jnp.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=jnp.float32)
	ky = jnp.transpose(kx)  # [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
	return kx, ky


def sobel_x(A, kx):
	"""Compute horizontal Sobel filter per channel."""
	# Apply convolve2d to each channel using vmap
	return jax.vmap(lambda a: convolve2d(a, kx, mode="same"), in_axes=2, out_axes=2)(A)


def sobel_y(A, ky):
	"""Compute vertical Sobel filter per channel."""
	return jax.vmap(lambda a: convolve2d(a, ky, mode="same"), in_axes=2, out_axes=2)(A)


def sobel(A):
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
