"""Dominated Novelty Search module.

Dominated Novelty Search (DNS) is a quality-diversity algorithm implementing local
competition through a dynamic fitness transformation: each individual is scored by its
mean descriptor-space distance to its k nearest *fitter* neighbors. This rewards
individuals that are either the fittest, or occupy unique regions of descriptor space
relative to better-performing solutions.

References:
	[1] Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity,
		Bahlous-Boldi et al. 2025.

"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from .complex_system import Genotype


def dominated_novelty_fn(fitness: Array, descriptor: Array, *, k: int) -> Array:
	"""Compute dominated novelty for a population.

	Args:
		fitness: Array with shape `(N,)`. Higher is better, invalid is `-inf`.
		descriptor: Array with shape `(N, D)`.
		k: Number of nearest fitter neighbors.

	Returns:
		Dominated novelty with shape `(N,)`. Champions (individuals with no fitter
			neighbor) score `+inf` and are always kept.

	"""
	valid = fitness != -jnp.inf

	# Neighbors
	neighbor = valid[:, None] & valid[None, :]
	neighbor = jnp.fill_diagonal(neighbor, False, inplace=False)

	# Fitter neighbors
	fitter = jnp.where(neighbor, fitness[None, :] >= fitness[:, None], False)

	# Distance to fitter neighbors
	distance = jnp.linalg.norm(descriptor[:, None, :] - descriptor[None, :, :], axis=-1)
	distance_fitter = jnp.where(fitter, distance, jnp.inf)

	# Mean distance to the k nearest fitter neighbors
	values, indices = jax.vmap(partial(jax.lax.top_k, k=k))(-distance_fitter)
	novelty = jnp.mean(-values, axis=-1, where=jnp.take_along_axis(fitter, indices, axis=-1))

	return jnp.where(jnp.isnan(novelty), jnp.inf, novelty)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DNSState:
	"""State of Dominated Novelty Search.

	Attributes:
		population: Genotype pytree with leading dimension `population_size`.
		fitness: Fitness with shape `(population_size,)`. Higher is better, invalid is
			`-inf`.
		descriptor: Descriptors with shape `(population_size, descriptor_size)`.
		observations: Observations of the population, retained for descriptor training and
			re-encoding.
		best_genotype: Fittest genotype found so far.
		best_fitness: Fitness of the best genotype.

	"""

	population: Genotype
	fitness: Array
	descriptor: Array
	observations: Array
	best_genotype: Genotype
	best_fitness: Array


class DNS:
	"""Dominated Novelty Search."""

	def __init__(
		self,
		*,
		sample_fn: Callable[[Array], Genotype],
		mutate_fn: Callable[[Array, Genotype], Genotype],
		population_size: int,
		num_children: int,
		k: int = 3,
		sample_ratio: float = 0.0,
	):
		"""Initialize DNS.

		Args:
			sample_fn: Samples a fresh genotype from the prior.
			mutate_fn: Mutates a parent genotype.
			population_size: Size of the population.
			num_children: Number of child individuals per generation.
			k: Number of nearest fitter neighbors for local competition.
			sample_ratio: Fraction of child individuals drawn fresh from `sample_fn`
				(`sampled_individuals`); the rest are `mutated_individuals`. 0.0 is pure
				mutation, 1.0 is pure random search.

		"""
		self.sample_fn = sample_fn
		self.mutate_fn = mutate_fn
		self.population_size = population_size
		self.num_children = num_children
		self.k = k
		self.num_samples = round(sample_ratio * num_children)

	def init(
		self,
		population: Genotype,
		fitness: Array,
		descriptor: Array,
		observations: Array,
	) -> DNSState:
		"""Initialize the DNS state from an evaluated population of any size.

		A population larger than `population_size` (an oversampled initialization) is
		downselected under dominated novelty competition; a smaller one is padded with
		invalid placeholders (fitness `-inf`) — never selected as parents, replaced by
		child_individuals over time — so a single living solution can seed the whole
		population.
		"""
		pad = self.population_size - fitness.shape[0]
		if pad > 0:

			def pad_like(x: Array) -> Array:
				return jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad, *x.shape[1:]))])

			population = jax.tree.map(pad_like, population)
			descriptor = pad_like(descriptor)
			observations = pad_like(observations)
			fitness = jnp.concatenate([fitness, jnp.full((pad,), -jnp.inf)])

		population, fitness, descriptor, observations = self._select(
			population, fitness, descriptor, observations
		)
		best_idx = jnp.argmax(fitness)
		return DNSState(
			population=population,
			fitness=fitness,
			descriptor=descriptor,
			observations=observations,
			best_genotype=jax.tree.map(lambda x: x[best_idx], population),
			best_fitness=fitness[best_idx],
		)

	def ask(self, key: Array, state: DNSState) -> Genotype:
		"""Produce child individuals: mutated parent individuals plus fresh samples."""
		key_select, key_mutate, key_sample = jax.random.split(key, 3)
		num_mutated = self.num_children - self.num_samples

		child_individuals = []
		if num_mutated > 0:
			valid = state.fitness != -jnp.inf
			p = jnp.where(jnp.any(valid), valid / jnp.sum(valid), 1.0 / valid.shape[0])
			indices = jax.random.choice(
				key_select, state.fitness.shape[0], shape=(num_mutated,), p=p
			)
			parent_individuals = jax.tree.map(lambda x: x[indices], state.population)

			keys = jax.random.split(key_mutate, num_mutated)
			mutated_individuals = jax.vmap(self.mutate_fn)(keys, parent_individuals)
			child_individuals.append(mutated_individuals)

		if self.num_samples > 0:
			keys = jax.random.split(key_sample, self.num_samples)
			sampled_individuals = jax.vmap(self.sample_fn)(keys)
			child_individuals.append(sampled_individuals)

		return jax.tree.map(lambda *x: jnp.concatenate(x), *child_individuals)

	def tell(
		self,
		state: DNSState,
		child_individuals: Genotype,
		fitness: Array,
		descriptor: Array,
		observations: Array,
	) -> DNSState:
		"""Apply mu+lambda selection under dominated novelty competition."""
		new_population, new_fitness, new_descriptor, new_observations = self._select(
			jax.tree.map(lambda x, y: jnp.concatenate([x, y]), state.population, child_individuals),
			jnp.concatenate([state.fitness, fitness]),
			jnp.concatenate([state.descriptor, descriptor]),
			jnp.concatenate([state.observations, observations]),
		)

		# Track the best genotype ever seen
		best_idx = jnp.argmax(new_fitness)
		is_better = new_fitness[best_idx] > state.best_fitness
		best_genotype = jax.tree.map(
			lambda new, old: jnp.where(is_better, new[best_idx], old),
			new_population,
			state.best_genotype,
		)

		return DNSState(
			population=new_population,
			fitness=new_fitness,
			descriptor=new_descriptor,
			observations=new_observations,
			best_genotype=best_genotype,
			best_fitness=jnp.maximum(state.best_fitness, new_fitness[best_idx]),
		)

	def _select(
		self,
		population: Genotype,
		fitness: Array,
		descriptor: Array,
		observations: Array,
	) -> tuple[Genotype, Array, Array, Array]:
		"""Keep the top `population_size` individuals by dominated novelty."""
		dominated_novelty = dominated_novelty_fn(fitness, descriptor, k=self.k)
		meta_fitness = jnp.where(fitness != -jnp.inf, dominated_novelty, -jnp.inf)
		indices = jnp.argsort(meta_fitness, descending=True)[: self.population_size]

		return (
			jax.tree.map(lambda x: x[indices], population),
			fitness[indices],
			descriptor[indices],
			observations[indices],
		)


class QDConfig(BaseModel):
	"""Quality-diversity search configuration; `build` constructs the `DNS` algorithm.

	`num_init`, `minibatch_size`, and `num_generations` configure the search loop around
	the algorithm (initialization size, evaluation memory bound, budget).
	"""

	model_config = ConfigDict(frozen=True)

	population_size: PositiveInt = 1024
	num_children: PositiveInt = 256
	k: PositiveInt = 3
	sample_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
	num_init: PositiveInt = 1
	minibatch_size: PositiveInt = 256
	num_generations: PositiveInt = 1024

	def build(
		self,
		sample_fn: Callable[[Array], Genotype],
		mutate_fn: Callable[[Array, Genotype], Genotype],
	) -> DNS:
		"""Build the configured search algorithm over a genotype space."""
		return DNS(
			sample_fn=sample_fn,
			mutate_fn=mutate_fn,
			population_size=self.population_size,
			num_children=self.num_children,
			k=self.k,
			sample_ratio=self.sample_ratio,
		)
