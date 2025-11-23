"""Tests for Flow Lenia."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.flow_lenia import FlowLenia
from cax.cs.lenia import LeniaRuleParams
from cax.cs.lenia.growth import GrowthParams
from cax.cs.lenia.kernel import KernelParams


def test_flow_lenia_jit_init() -> None:
	"""Test that FlowLenia can be instantiated under jax.jit."""

	@jax.jit
	def init_flow_lenia() -> FlowLenia:
		kernel_params = KernelParams(r=jnp.array([1.0]), b=jnp.array([[1.0]]))
		growth_params = GrowthParams(mean=jnp.array([0.5]), std=jnp.array([0.1]))
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
