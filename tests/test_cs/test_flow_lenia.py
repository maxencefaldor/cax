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


def test_flow_lenia_rejects_non_2d() -> None:
	"""Test that a non-2D spatial_dims is refused at construction."""
	kernel_params = LeniaKernelParams(r=jnp.array([1.0]), beta=jnp.array([[1.0]]))
	growth_params = LeniaGrowthParams(mean=jnp.array([0.5]), std=jnp.array([0.1]))
	rule_params = LeniaRuleParams(
		channel_source=jnp.array([0]),
		channel_target=jnp.array([0]),
		weight=jnp.array([1.0]),
		kernel_params=kernel_params,
		growth_params=growth_params,
	)
	with pytest.raises(ValueError, match="2 spatial dimensions"):
		FlowLenia(spatial_dims=(16, 16, 16), channel_size=1, R=5, T=10, rule_params=rule_params)


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
