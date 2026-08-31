"""Lenia mutation module."""

import jax
import jax.numpy as jnp
from jax import Array

from cax.cs.lenia import LeniaGrowthParams, LeniaKernelParams, LeniaRuleParams

from ...core.variation import mutate_bounded, reflect
from .config import LeniaConfig
from .genotype import Genotype


def mutate(key: Array, genotype: Genotype, config: LeniaConfig) -> Genotype:
	"""Mutate a genotype with per-parameter constraints."""
	key_weight, key_r, key_beta, key_mean, key_std, key_state = jax.random.split(key, 6)
	rule_params = genotype.rule_params
	mutation_std = config.mutate.mutation_std

	# Weights: Dirichlet perturbation (stays on the simplex). The floor pseudo-count
	# keeps every rule resurrectable; without it the perturbation is pure drift, whose
	# absorbing states are the simplex corners (see MutateConfig.weight_floor)
	weight = jax.random.dirichlet(
		key_weight,
		alpha=config.mutate.weight_concentration * rule_params.weight + config.mutate.weight_floor,
	)

	# Kernel radius: bounded Gaussian
	r = mutate_bounded(
		key_r,
		rule_params.kernel_params.r,
		lower=config.kernel_r_range[0],
		upper=config.kernel_r_range[1],
		mutation_std=mutation_std,
	)

	# Ring heights: Gaussian, reflected at 0, renormalized so the peak is 1.0
	beta = rule_params.kernel_params.beta
	is_inactive = jnp.isnan(beta)
	beta = beta + mutation_std * jax.random.normal(key_beta, beta.shape)
	beta = jnp.where(is_inactive, jnp.nan, reflect(beta, lower=0.0))
	beta = beta / jnp.nanmax(beta, axis=-1, keepdims=True)

	# Growth mean and std: bounded Gaussian
	mean = mutate_bounded(
		key_mean,
		rule_params.growth_params.mean,
		lower=config.growth_mean_range[0],
		upper=config.growth_mean_range[1],
		mutation_std=mutation_std,
	)
	std = mutate_bounded(
		key_std,
		rule_params.growth_params.std,
		lower=config.growth_std_range[0],
		upper=config.growth_std_range[1],
		mutation_std=mutation_std,
	)

	# Initial state: bounded Gaussian. Two alternatives were tried and rejected by eye
	# (2026-08-31): freezing the seed entirely, and support-preserving multiplicative
	# noise. Both looked worse than this operator despite measuring competitively
	state_init = mutate_bounded(
		key_state,
		genotype.state_init,
		lower=0.0,
		upper=1.0,
		mutation_std=mutation_std,
	)

	return Genotype(
		rule_params=LeniaRuleParams(
			channel_source=rule_params.channel_source,
			channel_target=rule_params.channel_target,
			weight=weight,
			kernel_params=LeniaKernelParams(r=r, beta=beta),
			growth_params=LeniaGrowthParams(mean=mean, std=std),
		),
		state_init=state_init,
	)
