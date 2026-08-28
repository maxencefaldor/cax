"""Tests for gradient flow through Lenia.

These pin down the differentiability facts that `examples/55_lenia_grad.ipynb` relies on:
gradients flow through entire rollouts, normalized parameters are flat along their scale
direction, and the kernel radius is differentiable exactly when the kernel core vanishes
at its support boundary.
"""

import jax
import jax.core
import jax.numpy as jnp

from cax.cs.lenia import Lenia, LeniaRuleParams
from cax.cs.lenia.growth import LeniaGrowthParams
from cax.cs.lenia.kernel import (
	FreeKernelParams,
	LeniaKernelParams,
	exponential_kernel_core,
	exponential_kernel_fn,
	free_kernel_fn,
	gaussian_kernel_core,
	polynomial_kernel_core,
)


def make_rule(num_kernels: int = 1) -> LeniaRuleParams:
	"""Build a small single-channel rule with `num_kernels` rank-1 kernels."""
	return LeniaRuleParams(
		channel_source=jnp.zeros((num_kernels,), jnp.int32),
		channel_target=jnp.zeros((num_kernels,), jnp.int32),
		weight=jnp.linspace(1.0, 2.0, num_kernels),
		kernel_params=LeniaKernelParams(
			r=jnp.linspace(0.7, 1.0, num_kernels),
			beta=jnp.ones((num_kernels, 1)),
		),
		growth_params=LeniaGrowthParams(
			mean=jnp.full((num_kernels,), 0.15),
			std=jnp.full((num_kernels,), 0.02),
		),
	)


def make_state(key: jax.Array) -> jax.Array:
	"""Sample a smooth blob of mass at the center of a small grid."""
	state = jnp.zeros((32, 32, 1))

	return state.at[12:20, 12:20].set(jax.random.uniform(key, (8, 8, 1)))


def rollout_loss(
	rule_params: LeniaRuleParams, state: jax.Array, *, num_steps: int = 8, **kwargs
) -> jax.Array:
	"""Compute the mean state after a short rollout, building the system inside."""
	lenia = Lenia(
		spatial_dims=(32, 32), channel_size=1, R=5, T=10, rule_params=rule_params, **kwargs
	)

	def step_fn(state: jax.Array, _: None) -> tuple[jax.Array, None]:
		return lenia.update(state, lenia.perceive(state)), None

	return jnp.mean(jax.lax.scan(step_fn, state, length=num_steps)[0])


def float_and_static(rule_params: LeniaRuleParams) -> tuple[LeniaRuleParams, LeniaRuleParams]:
	"""Split a rule into its float leaves and its integer leaves."""

	def is_inexact(leaf: jax.Array) -> bool:
		return jnp.issubdtype(leaf.dtype, jnp.inexact)

	params = jax.tree.map(lambda x: x if is_inexact(x) else None, rule_params)
	static = jax.tree.map(lambda x: None if is_inexact(x) else x, rule_params)

	return params, static


def test_scan_rollout_matches_driver_bitwise() -> None:
	"""Test that scanning perceive/update reproduces the driver's states exactly."""
	rule_params = make_rule()
	state = make_state(jax.random.key(0))
	lenia = Lenia(spatial_dims=(32, 32), channel_size=1, R=5, T=10, rule_params=rule_params)

	def step_fn(state: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
		next_state = lenia.update(state, lenia.perceive(state))
		return next_state, next_state

	final_scan, states_scan = jax.lax.scan(step_fn, state, length=8)
	final_driver, states_driver = lenia(state, num_steps=8, return_states=True)

	assert jnp.array_equal(final_scan, final_driver)
	assert jnp.array_equal(states_scan, states_driver)


def test_gradient_flows_through_rollout() -> None:
	"""Test that the gradient of a rollout is finite, and nonzero for the growth params."""
	params, static = float_and_static(make_rule())
	state = make_state(jax.random.key(0))

	def loss_fn(params: LeniaRuleParams) -> jax.Array:
		return rollout_loss(
			jax.tree.map(
				lambda p, s: s if p is None else p, params, static, is_leaf=lambda x: x is None
			),
			state,
		)

	grads = jax.grad(loss_fn)(params)

	assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))
	assert jnp.linalg.norm(grads.growth_params.mean) > 0.0
	assert jnp.linalg.norm(grads.growth_params.std) > 0.0


def test_normalized_parameters_have_zero_scale_gradient() -> None:
	"""Test Euler's relation: the gradient is orthogonal to each normalized parameter.

	The kernel weights are divided by their sum and the kernel by its own integral, so the
	loss is scale-free in `weight` and in each kernel's `beta`. For a degree-0 homogeneous
	function, Euler's theorem forces `sum_k theta_k * dL/dtheta_k = 0`.
	"""
	rule_params = make_rule(num_kernels=3)
	params, static = float_and_static(rule_params)
	state = make_state(jax.random.key(0))

	def loss_fn(params: LeniaRuleParams) -> jax.Array:
		return rollout_loss(
			jax.tree.map(
				lambda p, s: s if p is None else p, params, static, is_leaf=lambda x: x is None
			),
			state,
		)

	grads = jax.grad(loss_fn)(params)

	# The repartition between kernels carries gradient; only the overall scale is flat.
	assert jnp.linalg.norm(grads.weight) > 0.0
	scale_component = jnp.sum(rule_params.weight * grads.weight)
	assert jnp.abs(scale_component) < 1e-6 * jnp.linalg.norm(grads.weight)

	beta_scale = jnp.sum(rule_params.kernel_params.beta * grads.kernel_params.beta, axis=-1)
	assert jnp.all(jnp.abs(beta_scale) < 1e-6)


def test_kernel_cores_at_the_support_boundary() -> None:
	"""Test which cores vanish where the support mask cuts them.

	The kernel is `mask * beta[segment] * core(position)` with a hard mask `radius < r`,
	so `r` is differentiable exactly when the core is zero at the boundary. The Gaussian
	core is the exception, which is why the gradient of `r` under it is untrustworthy.
	"""
	edge = jnp.array(1.0)

	assert exponential_kernel_core(edge) == 0.0
	assert polynomial_kernel_core(edge) == 0.0
	assert gaussian_kernel_core(edge) > 0.0


def test_radius_gradient_matches_finite_difference_under_exponential_core() -> None:
	"""Test that the autodiff gradient of the kernel radius is a real derivative."""
	rule_params = make_rule()
	params, static = float_and_static(rule_params)
	state = make_state(jax.random.key(0))

	def loss_of_r(r: jax.Array) -> jax.Array:
		bumped = LeniaRuleParams(
			channel_source=static.channel_source,
			channel_target=static.channel_target,
			weight=params.weight,
			kernel_params=LeniaKernelParams(r=r, beta=params.kernel_params.beta),
			growth_params=params.growth_params,
		)
		return rollout_loss(bumped, state, kernel_fn=exponential_kernel_fn)

	r = rule_params.kernel_params.r
	autodiff = jax.grad(lambda r: loss_of_r(r))(r)[0]
	eps = 1e-3
	finite_difference = (loss_of_r(r + eps) - loss_of_r(r - eps)) / (2 * eps)

	assert jnp.abs(autodiff) > 0.0
	assert jnp.abs(finite_difference - autodiff) < 0.2 * jnp.abs(autodiff)


def test_off_band_potential_collapses_the_rule_gradient() -> None:
	"""Test that a state far from the growth band leaves the rule almost invisible.

	The growth mapping is a Gaussian bell. A uniform state drives the potential to about
	0.5, many standard deviations from a growth mean of 0.15, so the bell is numerically
	flat there and the rule parameters — which reach the loss only through it — collapse
	to a gradient some fifteen orders of magnitude below their in-band value. In a narrow
	enough band the bell underflows and the gradient is exactly zero.

	The state gradient survives either way, because the update adds growth to the state
	the cell already had, and that path does not pass through the bell.
	"""
	rule_params = make_rule()
	params, static = float_and_static(rule_params)
	state = jax.random.uniform(jax.random.key(0), (32, 32, 1))

	def merge(params: LeniaRuleParams) -> LeniaRuleParams:
		return jax.tree.map(
			lambda p, s: s if p is None else p, params, static, is_leaf=lambda x: x is None
		)

	def rule_gradient_norm(num_steps: int) -> jax.Array:
		grads = jax.grad(lambda params: rollout_loss(merge(params), state, num_steps=num_steps))(
			params
		)
		return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(grads)))

	# One step in, the potential is still off-band and the rule is invisible.
	assert rule_gradient_norm(1) < 1e-12

	# By the fourth step the state has relaxed into the band and the rule matters again.
	assert rule_gradient_norm(4) > 1e-2

	state_grads = jax.grad(lambda state: rollout_loss(rule_params, state, num_steps=1))(state)
	assert jnp.linalg.norm(state_grads) > 1e-3


def test_free_kernel_is_invariant_under_its_scale_gauge() -> None:
	"""Test that (r, a, w) -> (s * r, a / s, w / s) is the same free kernel.

	`free_kernel_fn` places a bump at absolute radius `a * r` with width `w * r`, so the
	three parameters carry only two degrees of freedom. Fitting a trajectory can recover
	`a * r` and `w * r` but never `r` on its own.
	"""
	radius = jnp.linspace(0.0, 1.2, 256)
	kernel_params = FreeKernelParams(
		r=jnp.array(0.9),
		b=jnp.array([1.0, 0.4]),
		a=jnp.array([0.5, 0.8]),
		w=jnp.array([0.15, 0.05]),
	)
	kernel = free_kernel_fn(radius, kernel_params)

	for scale in (0.7, 1.4):
		rescaled = FreeKernelParams(
			r=kernel_params.r * scale,
			b=kernel_params.b,
			a=kernel_params.a / scale,
			w=kernel_params.w / scale,
		)
		assert jnp.allclose(free_kernel_fn(radius, rescaled), kernel, atol=1e-6)


def test_building_a_system_under_jit_leaves_the_rule_intact() -> None:
	"""Test that a traced call does not write tracers into the caller's rule.

	A module that stores a caller-supplied parameter object is holding part of the
	caller's state, and flax writes a module's state back after a transformed call. An
	aliased container built inside `jax.jit` therefore has tracers written into it,
	corrupting the rule for every later use and leaking a tracer out of the trace.
	"""
	from cax.cs.lenia import load_pattern

	pattern, rule_params = load_pattern("Orbium")
	state = jnp.zeros((32, 32, 1)).at[6:26, 6:26].set(pattern)

	@jax.jit
	def loss(state: jax.Array) -> jax.Array:
		lenia = Lenia(spatial_dims=(32, 32), channel_size=1, R=13, T=10, rule_params=rule_params)
		return jnp.sum(lenia(state, num_steps=4))

	loss(state)

	for leaf in jax.tree.leaves(rule_params):
		assert not isinstance(leaf, jax.core.Tracer)


def test_the_driver_composes_with_nested_transformations() -> None:
	"""Test that `jax.jvp` still works after `jax.jit` of `jax.grad` over the driver.

	Writing tracers into a shared rule used to make a later, unrelated transformation
	fail with an `UnexpectedTracerError`. That failure was order dependent and so is not
	itself a reliable regression test — `test_building_a_system_under_jit_leaves_the_rule_intact`
	covers the cause. This covers the pattern end to end.
	"""
	from cax.cs.lenia import load_pattern

	pattern, rule_params = load_pattern("Orbium")
	state = jnp.zeros((64, 64, 1)).at[22:42, 22:42].set(pattern)

	def build() -> Lenia:
		return Lenia(spatial_dims=(64, 64), channel_size=1, R=13, T=10, rule_params=rule_params)

	@jax.jit
	def descend(state: jax.Array) -> jax.Array:
		return state - 0.01 * jax.grad(lambda s: jnp.sum(build()(s, num_steps=16)))(state)

	for _ in range(2):
		state = descend(state)

	jax.jvp(
		lambda s: build()(s, num_steps=16, return_states=True)[1],
		(state,),
		(jnp.ones_like(state),),
	)
