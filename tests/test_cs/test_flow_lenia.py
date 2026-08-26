"""Tests for Flow Lenia."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.flow_lenia import FlowLenia
from cax.cs.lenia import LeniaRuleParams
from cax.cs.lenia.growth import LeniaGrowthParams
from cax.cs.lenia.kernel import LeniaKernelParams


def test_flow_lenia_jit_init() -> None:
	"""Test that FlowLenia can be instantiated under jax.jit."""

	@jax.jit
	def init_flow_lenia() -> FlowLenia:
		kernel_params = LeniaKernelParams(r=jnp.array([1.0]), beta=jnp.array([[1.0]]))
		growth_params = LeniaGrowthParams(mean=jnp.array([0.5]), std=jnp.array([0.1]))
		rule_params = LeniaRuleParams(
			channel_source=jnp.array([0]),
			channel_target=jnp.array([0]),
			weight=jnp.array([1.0]),
			kernel_params=kernel_params,
			growth_params=growth_params,
		)
		flow_lenia = FlowLenia(
			spatial_dims=(32, 32),
			channel_size=1,
			R=5,
			T=10,
			rule_params=rule_params,
		)
		return flow_lenia

	try:
		init_flow_lenia()
	except Exception as e:
		pytest.fail(f"FlowLenia instantiation failed under jit: {e}")


def single_channel_rule_params(mean: float = 0.15, std: float = 0.015) -> LeniaRuleParams:
	"""Return single-channel, single-kernel rule parameters."""
	return LeniaRuleParams(
		channel_source=jnp.array([0]),
		channel_target=jnp.array([0]),
		weight=jnp.array([1.0]),
		kernel_params=LeniaKernelParams(r=jnp.array([1.0]), beta=jnp.array([[1.0]])),
		growth_params=LeniaGrowthParams(mean=jnp.array([mean]), std=jnp.array([std])),
	)


@pytest.mark.parametrize("spatial_dims", [(48,), (16, 16), (10, 10, 10)])
def test_flow_lenia_conserves_mass_in_any_dimension(spatial_dims: tuple[int, ...]) -> None:
	"""Test that mass is conserved and stays non-negative in 1D, 2D, and 3D."""
	flow_lenia = FlowLenia(
		spatial_dims=spatial_dims,
		channel_size=1,
		R=5,
		T=5,
		rule_params=single_channel_rule_params(),
		dd=2,
	)

	key = jax.random.key(0)
	state = jax.random.uniform(key, (*spatial_dims, 1))
	mass_before = jnp.sum(state)

	state_final = flow_lenia(state, num_steps=20)

	assert jnp.allclose(mass_before, jnp.sum(state_final), rtol=1e-4)
	assert jnp.min(state_final) >= 0.0


def test_sobel_matches_analytic_gradient_in_3d() -> None:
	"""Test the 3D Sobel gradients against the analytic gradient of a linear field."""
	from cax.cs.flow_lenia.update import sobel

	spatial_dims = (10, 11, 12)
	coordinates = jnp.meshgrid(
		*[jnp.arange(dim, dtype=jnp.float32) for dim in spatial_dims], indexing="ij"
	)
	slopes = jnp.array([3.0, 5.0, -2.0])
	field = sum(slope * coordinate for slope, coordinate in zip(slopes, coordinates, strict=True))

	gradients = sobel(field[..., None])
	interior = tuple(slice(1, -1) for _ in spatial_dims)

	# The kernel is a derivative stencil [1, 0, -1] along its axis and a smoothing
	# stencil [1, 2, 1] along the others, so each component is scaled by 2 * 4^(D - 1).
	scale = 2 * 4 ** (len(spatial_dims) - 1)
	assert jnp.allclose(gradients[interior][..., 0], slopes * scale, atol=1e-3)


@pytest.mark.parametrize("spatial_dims", [(64,), (32, 47), (12, 13, 14)])
def test_sobel_commutes_with_translation(spatial_dims: tuple[int, ...]) -> None:
	"""Test the torus property: differentiation commutes with translation, bitwise.

	The official implementation zero-pads the Sobel while its perception and
	reintegration wrap; cax deliberately wraps the Sobel too (a stated deviation),
	and this exact equivariance is what that buys.
	"""
	from cax.cs.flow_lenia.update import sobel

	num_spatial_dims = len(spatial_dims)
	key = jax.random.key(0)
	field = jax.random.uniform(key, (*spatial_dims, 2))

	axes = tuple(range(num_spatial_dims))
	shift = tuple(range(1, num_spatial_dims + 1))
	assert jnp.array_equal(
		sobel(jnp.roll(field, shift, axis=axes)), jnp.roll(sobel(field), shift, axis=axes)
	)


def test_sobel_matches_official_zero_padded_gradients_in_the_interior() -> None:
	"""Test bitwise identity with the official convolve2d Sobel away from the boundary."""
	from jax.scipy.signal import convolve2d

	from cax.cs.flow_lenia.update import sobel

	key = jax.random.key(0)
	field = jax.random.uniform(key, (33, 47, 3))

	# The official implementation's Sobel: convolve2d(mode="same"), zero-padded.
	kx = jnp.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
	ky = jnp.transpose(kx)

	def per_channel(a, k):
		return jax.vmap(
			lambda channel: convolve2d(channel, k, mode="same"), in_axes=-1, out_axes=-1
		)(a)

	official = jnp.stack([per_channel(field, ky), per_channel(field, kx)], axis=-2)

	interior = (slice(1, -1), slice(1, -1))
	assert jnp.array_equal(sobel(field)[interior], official[interior])


def test_flow_lenia_is_isotropic_in_3d() -> None:
	"""Test that a radially symmetric blob stays symmetric under symmetric flow."""
	flow_lenia = FlowLenia(
		spatial_dims=(16, 16, 16),
		channel_size=1,
		R=5,
		T=5,
		rule_params=single_channel_rule_params(),
		dd=2,
	)

	axis = jnp.arange(16.0) - 7.5
	radius = jnp.linalg.norm(
		jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1), axis=-1
	)
	state = jnp.exp(-(radius**2) / 8.0)[..., None]

	state_final = flow_lenia(state, num_steps=5)

	# The dynamics are exactly isotropic; the residual is float32 FFT roundoff, which
	# is ~1e-5 relative here (~1e-15 in float64) — orders of magnitude below any real
	# axis bias, which would show up at the scale of the state itself.
	for axes in [(1, 0, 2, 3), (2, 1, 0, 3), (0, 2, 1, 3)]:
		assert jnp.allclose(state_final, jnp.transpose(state_final, axes), atol=1e-4)
	for spatial_axis in range(3):
		assert jnp.allclose(state_final, jnp.flip(state_final, axis=spatial_axis), atol=1e-4)


def test_flow_lenia_theta_a_defaults_to_channel_size() -> None:
	"""Test that theta_A defaults to channel_size, the official implementation's value."""
	rule_params = LeniaRuleParams(
		channel_source=jnp.array([0, 1, 2]),
		channel_target=jnp.array([0, 1, 2]),
		weight=jnp.array([1.0, 1.0, 1.0]),
		kernel_params=LeniaKernelParams(
			r=jnp.array([1.0, 1.0, 1.0]), beta=jnp.array([[1.0], [1.0], [1.0]])
		),
		growth_params=LeniaGrowthParams(
			mean=jnp.array([0.5, 0.5, 0.5]), std=jnp.array([0.1, 0.1, 0.1])
		),
	)
	flow_lenia = FlowLenia(
		spatial_dims=(32, 32), channel_size=3, R=5, T=10, rule_params=rule_params
	)
	assert flow_lenia.update.theta_A == 3

	flow_lenia_explicit = FlowLenia(
		spatial_dims=(32, 32), channel_size=3, R=5, T=10, rule_params=rule_params, theta_A=2.5
	)
	assert flow_lenia_explicit.update.theta_A == 2.5


def test_flow_lenia_conserves_mass() -> None:
	"""Test that total mass is conserved over many steps up to float32 accumulation."""
	kernel_params = LeniaKernelParams(r=jnp.array([1.0]), beta=jnp.array([[1.0]]))
	growth_params = LeniaGrowthParams(mean=jnp.array([0.15]), std=jnp.array([0.015]))
	rule_params = LeniaRuleParams(
		channel_source=jnp.array([0]),
		channel_target=jnp.array([0]),
		weight=jnp.array([1.0]),
		kernel_params=kernel_params,
		growth_params=growth_params,
	)
	flow_lenia = FlowLenia(
		spatial_dims=(64, 64), channel_size=1, R=10, T=5, rule_params=rule_params
	)

	state = jnp.zeros((64, 64, 1))
	patch = jax.random.uniform(jax.random.key(2), (20, 20, 1))
	state = state.at[22:42, 22:42].set(patch)

	mass_before = jnp.sum(state)
	state_final = flow_lenia(state, num_steps=100)
	mass_after = jnp.sum(state_final)

	# The algorithm conserves mass exactly; float32 summation drifts ~1.5e-7 per step.
	assert jnp.allclose(mass_before, mass_after, rtol=1e-4)
	assert jnp.min(state_final) >= 0.0
