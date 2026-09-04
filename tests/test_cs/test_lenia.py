"""Tests for Lenia."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.lenia import Lenia, LeniaRuleParams
from cax.cs.lenia.growth import LeniaGrowthParams
from cax.cs.lenia.kernel import LeniaKernelParams


def test_lenia_jit_init() -> None:
    """Test that Lenia can be instantiated under jax.jit."""

    @jax.jit
    def init_lenia() -> Lenia:
        kernel_params = LeniaKernelParams(r=jnp.array([1.0]), beta=jnp.array([[1.0]]))
        growth_params = LeniaGrowthParams(mean=jnp.array([0.5]), std=jnp.array([0.1]))
        rule_params = LeniaRuleParams(
            channel_source=jnp.array([0]),
            channel_target=jnp.array([0]),
            weight=jnp.array([1.0]),
            kernel_params=kernel_params,
            growth_params=growth_params,
        )
        lenia = Lenia(
            spatial_dims=(32, 32),
            channel_size=1,
            R=5,
            T=10,
            rule_params=rule_params,
        )
        return lenia

    try:
        init_lenia()
    except Exception as e:
        pytest.fail(f"Lenia instantiation failed under jit: {e}")


def test_load_pattern_all_shipped_names() -> None:
    """Test that every shipped pattern loads with consistent shapes."""
    from cax.cs.lenia import PATTERN_NAMES, load_pattern

    for name in PATTERN_NAMES:
        pattern, rule_params = load_pattern(name)
        num_kernels = rule_params.weight.shape[0]
        assert pattern.ndim == 3
        assert rule_params.channel_source.shape == (num_kernels,)
        assert rule_params.channel_target.shape == (num_kernels,)
        assert rule_params.kernel_params.r.shape == (num_kernels,)
        assert rule_params.growth_params.mean.shape == (num_kernels,)


def test_load_pattern_unknown_name() -> None:
    """Test that an unknown pattern name is refused with the catalogue listed."""
    from cax.cs.lenia import load_pattern

    with pytest.raises(ValueError, match="Shipped patterns"):
        load_pattern("Gliderium")


def test_orbium_survives() -> None:
    """Test that the shipped Orbium survives 200 steps.

    It should stay localized and keep a near-constant mass.
    """
    from cax.cs.lenia import load_pattern, metrics_fn

    pattern, rule_params = load_pattern("Orbium")
    R, T = 13, 10
    lenia = Lenia(
        spatial_dims=(64, 64), channel_size=1, R=R, T=T, rule_params=rule_params
    )

    state = jnp.zeros((64, 64, 1)).at[22:42, 22:42].set(pattern)
    mass_before = metrics_fn(state, R=R)["mass"]

    state_final = lenia(state, num_steps=200)
    metrics_final = metrics_fn(state_final, R=R)

    assert jnp.allclose(metrics_final["mass"], mass_before, rtol=0.1)
    assert metrics_final["concentration"] > 0.5


def test_lenia_runs_in_3d() -> None:
    """Test that Lenia constructs, steps, and measures in three spatial dimensions."""
    from cax.cs.lenia import load_pattern, metrics_fn

    _, rule_params = load_pattern("Orbium")
    lenia = Lenia(
        spatial_dims=(16, 16, 16), channel_size=1, R=5, T=10, rule_params=rule_params
    )

    key = jax.random.key(0)
    state = jax.random.uniform(key, (16, 16, 16, 1))
    state_final, states = lenia(state, num_steps=4, return_states=True)

    assert state_final.shape == (16, 16, 16, 1)
    assert states.shape == (4, 16, 16, 16, 1)
    metrics = metrics_fn(state_final, R=5)
    assert metrics["center_of_mass"].shape == (3,)
