"""Lenia sampling module.

Genotype sampling strategies. Every strategy has the same signature
`(key, config) -> Genotype` and is selected by name through `config.sample.strategy`, so
adding a strategy (e.g. a learned initial state) is one function and one registry entry:

- `noise`: random rules, centered uniform-noise blob — de-novo discovery.
- `soliton`: random rules, the pattern of a known soliton.
- `soliton_full`: a known soliton's own catalogued rules *and* pattern — breeding from a
  lineage, as in the official Leniabreeder.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import Array

from cax.cs.lenia import LeniaGrowthParams, LeniaKernelParams, LeniaRuleParams, load_pattern

from .config import LeniaConfig
from .genotype import Genotype


def sample(key: Array, config: LeniaConfig) -> Genotype:
	"""Sample a fresh genotype using the configured strategy."""
	return SAMPLE_FNS[config.sample.strategy](key, config)


def noise(key: Array, config: LeniaConfig) -> Genotype:
	"""Random rules and a centered uniform-noise blob."""
	key_rules, key_state = jax.random.split(key)
	num_spatial_dims = len(config.spatial_dims)
	blob = jax.random.uniform(
		key_state, (*(config.sample.blob_size,) * num_spatial_dims, config.channel_size)
	)
	return Genotype(
		rule_params=sample_rule_params(key_rules, config),
		state_init=_place(blob, config.spatial_dims),
	)


def soliton(key: Array, config: LeniaConfig) -> Genotype:
	"""Random rules and the pattern of one of the configured solitons."""
	key_rules, key_choice = jax.random.split(key)
	states = jnp.stack([_pattern_state(name, config) for name in config.sample.pattern_names])
	idx = jax.random.randint(key_choice, (), 0, len(config.sample.pattern_names))
	return Genotype(
		rule_params=sample_rule_params(key_rules, config),
		state_init=states[idx],
	)


def soliton_full(key: Array, config: LeniaConfig) -> Genotype:
	"""One of the configured solitons with its own catalogued rules and pattern.

	Note: the catalogued growth parameters must lie inside the configured mutation
	ranges, or reflection will fold them away from the creature (e.g. the Aquarium needs
	`growth_std_range` up to 0.18).
	"""
	genotypes = []
	for name in config.sample.pattern_names:
		_, rule_params = load_pattern(name)
		if rule_params.kernel_params.beta.shape[-1] != config.kernel_rank:
			raise ValueError(f"Pattern {name!r} does not match kernel_rank {config.kernel_rank}.")
		# Catalogued weights are raw growth heights; normalize onto the simplex
		rule_params = replace(rule_params, weight=rule_params.weight / jnp.sum(rule_params.weight))
		genotypes.append(Genotype(rule_params=rule_params, state_init=_pattern_state(name, config)))

	stacked = jax.tree.map(lambda *x: jnp.stack(x), *genotypes)
	idx = jax.random.randint(key, (), 0, len(genotypes))
	return jax.tree.map(lambda x: x[idx], stacked)


def sample_rule_params(key: Array, config: LeniaConfig) -> LeniaRuleParams:
	"""Sample random rule parameters from the prior."""
	key_weight, key_kernel, key_growth = jax.random.split(key, 3)

	channel_source, channel_target = channel_topology(config)
	num_kernels = channel_source.shape[0]

	# Weights jointly via Dirichlet (sum to 1 by construction)
	weight = jax.random.dirichlet(key_weight, alpha=jnp.ones(num_kernels))

	kernel_params = jax.vmap(lambda key: _sample_kernel_params(key, config))(
		jax.random.split(key_kernel, num_kernels)
	)
	growth_params = jax.vmap(lambda key: _sample_growth_params(key, config))(
		jax.random.split(key_growth, num_kernels)
	)

	return LeniaRuleParams(
		channel_source=channel_source,
		channel_target=channel_target,
		weight=weight,
		kernel_params=kernel_params,
		growth_params=growth_params,
	)


def channel_topology(config: LeniaConfig) -> tuple[Array, Array]:
	"""Structured channel connectivity (Chan 2020): self then cross kernels."""
	self_sources = jnp.repeat(jnp.arange(config.channel_size), config.num_self_kernels)

	cross_pairs = [
		(i, j) for i in range(config.channel_size) for j in range(config.channel_size) if i != j
	]
	# Explicit int dtype: at channel_size 1 the pair list is empty and jnp.array([])
	# would silently promote the whole topology to float
	cross_sources = jnp.repeat(
		jnp.array([i for i, _ in cross_pairs], dtype=jnp.int32), config.num_cross_kernels
	)
	cross_targets = jnp.repeat(
		jnp.array([j for _, j in cross_pairs], dtype=jnp.int32), config.num_cross_kernels
	)

	channel_source = jnp.concatenate([self_sources, cross_sources])
	channel_target = jnp.concatenate([self_sources, cross_targets])
	return channel_source, channel_target


def _sample_kernel_params(key: Array, config: LeniaConfig) -> LeniaKernelParams:
	"""Sample kernel parameters with a random number of active rings."""
	key_r, key_beta, key_rank, key_peak = jax.random.split(key, 4)

	r = jax.random.uniform(key_r, minval=config.kernel_r_range[0], maxval=config.kernel_r_range[1])

	# Rank: uniform over {1, ..., kernel_rank}; inactive rings are nan
	rank = jax.random.randint(key_rank, (), minval=1, maxval=config.kernel_rank + 1)
	active = jnp.arange(config.kernel_rank) < rank

	# Ring heights in [0, 1], with one active ring pinned at 1.0
	beta = jax.random.uniform(key_beta, (config.kernel_rank,))
	peak_idx = jax.random.randint(key_peak, (), minval=0, maxval=rank)
	beta = beta.at[peak_idx].set(1.0)
	beta = jnp.where(active, beta, jnp.nan)

	return LeniaKernelParams(r=r, beta=beta)


def _sample_growth_params(key: Array, config: LeniaConfig) -> LeniaGrowthParams:
	"""Sample growth parameters."""
	key_mean, key_std = jax.random.split(key)
	mean = jax.random.uniform(
		key_mean, minval=config.growth_mean_range[0], maxval=config.growth_mean_range[1]
	)
	std = jax.random.uniform(
		key_std, minval=config.growth_std_range[0], maxval=config.growth_std_range[1]
	)
	return LeniaGrowthParams(mean=mean, std=std)


def _pattern_state(name: str, config: LeniaConfig) -> Array:
	"""Load a shipped pattern and place it at the center of an empty state."""
	pattern, _ = load_pattern(name)
	if pattern.shape[-1] != config.channel_size:
		raise ValueError(f"Pattern {name!r} has {pattern.shape[-1]} channels.")
	for axis in range(len(config.spatial_dims)):
		pattern = pattern.repeat(config.state_scale, axis=axis)
	return _place(pattern, config.spatial_dims)


def _place(stamp: Array, spatial_dims: tuple[int, ...]) -> Array:
	"""Place a stamp at the center of an empty state."""
	state = jnp.zeros((*spatial_dims, stamp.shape[-1]))
	starts = tuple(
		(dim - size) // 2 for dim, size in zip(spatial_dims, stamp.shape[:-1], strict=True)
	)
	return jax.lax.dynamic_update_slice(state, stamp, (*starts, 0))


SAMPLE_FNS = {
	"noise": noise,
	"soliton": soliton,
	"soliton_full": soliton_full,
}
