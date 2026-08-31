"""Lenia configuration."""

from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict

from breeder.core import ComplexSystem


class SampleConfig(BaseModel):
	"""Configuration of genotype sampling.

	Attributes:
		strategy: Sampling strategy (see `lenia.sample`).
		blob_size: Side length of the noise blob (`noise` strategy).
		pattern_names: Shipped patterns to sample from (`soliton*` strategies).

	"""

	model_config = ConfigDict(frozen=True)

	strategy: Literal["noise", "soliton", "soliton_full"] = "soliton_full"
	blob_size: int = 24
	pattern_names: tuple[str, ...] = ("5N7KKM",)


class MutateConfig(BaseModel):
	"""Configuration of genotype mutation.

	Attributes:
		mutation_std: Mutation standard deviation, relative to each parameter's range.
		weight_concentration: Dirichlet concentration for weight mutation.
		weight_floor: Pseudo-count added to the Dirichlet alphas. At 0 the perturbation
			is pure drift and fixates — measured: 6 of 8 rules dead (< 1e-6) after ~100
			applications, all but one by ~300, well within one run — because a weight
			near 0 gets alpha near 0, which is absorbing. The default 0.5
			(Jeffreys-style) keeps every rule resurrectable: zero dead rules after 1000
			applications. Adopted as the default 2026-08-31 on the user's visual verdict
			over two seeds. It costs a weak pull toward uniform (at w = 1, E[w'] ≈ 0.97),
			so an extreme weight vector must be held there by selection.
			`weight_concentration` was left at 100: raising it to 1600 for step-size
			parity with the other parameters measured *worse* on both seeds, so the
			search wants the larger weight steps.
		state_strategy: How the initial state mutates. `gaussian` is the draft operator
			and the default (bounded per-pixel Gaussian; its reflection at 0 makes empty
			pixels *gain* mass in expectation). `multiplicative` scales pixels by
			`exp(std * noise)` — support-preserving, empty pixels stay exactly empty; it
			was the only arm above base fitness on both seeds (0.0764/0.0830 vs base mean
			0.0684) and looked good, so it remains the candidate worth revisiting.
			`frozen` disables seed mutation: **rejected** 2026-08-31 by visual verdict
			despite holding the highest phenotype variance of any arm, which settles that
			seed evolution earns its 49k of the genotype's ~49k+30 dimensions.

	"""

	model_config = ConfigDict(frozen=True)

	mutation_std: float = 0.01
	weight_concentration: float = 100.0
	weight_floor: float = 0.5
	state_strategy: Literal["gaussian", "multiplicative", "frozen"] = "gaussian"


class LeniaConfig(BaseModel):
	"""Configuration of the Lenia complex system and its genotype space.

	The parameter ranges define the genotype space itself: sampling draws from them and
	mutation reflects at them.

	Attributes:
		name: Discriminator tag of the complex system.
		spatial_dims: Spatial dimensions of the world.
		channel_size: Number of channels.
		R: Space resolution defining the kernel radius.
		T: Time resolution, in CAX's weight-averaged growth convention.
		state_scale: Scaling factor applied to state values.
		num_steps: Number of simulation steps per development.
		num_self_kernels: Kernels per (channel, itself) pair.
		num_cross_kernels: Kernels per ordered pair of distinct channels.
		kernel_rank: Maximum number of kernel rings.
		kernel_r_range: Range of the kernel radius, relative to `R`.
		growth_mean_range: Range of the growth mean.
		growth_std_range: Range of the growth std.
		min_mass: Minimum mass at every step for a phenotype to be valid.
		min_concentration: Minimum spatial concentration at every step for a phenotype
			to be valid.
		sample: Genotype sampling configuration.
		mutate: Genotype mutation configuration.

	"""

	model_config = ConfigDict(frozen=True)

	name: Literal["lenia"] = "lenia"
	spatial_dims: tuple[int, int] = (128, 128)
	channel_size: int = 3
	R: int = 12
	T: float = 1 / 3
	state_scale: int = 1
	num_steps: int = 128
	num_self_kernels: int = 3
	num_cross_kernels: int = 1
	kernel_rank: int = 3

	kernel_r_range: tuple[float, float] = (0.1, 1.2)
	growth_mean_range: tuple[float, float] = (0.05, 0.7)
	# The Aquarium's catalogued growth stds reach 0.18; the range is wide enough that
	# reflection does not fold them away from the creature
	growth_std_range: tuple[float, float] = (0.005, 0.2)

	min_mass: float = 0.5
	min_concentration: float = 0.5

	sample: SampleConfig = SampleConfig()
	mutate: MutateConfig = MutateConfig()

	def build(self) -> ComplexSystem:
		"""Build the Lenia complex system bound to this config."""
		# Deferred import: the sibling modules import LeniaConfig at load time
		from breeder.cs.lenia import SERIES, develop, mutate, sample, valid

		return ComplexSystem(
			sample_fn=partial(sample, config=self),
			mutate_fn=partial(mutate, config=self),
			develop_fn=partial(develop, config=self),
			valid_fn=partial(valid, config=self),
			spatial_dims=self.spatial_dims,
			unit=float(self.R * self.state_scale),
			series=SERIES,
			num_frames=self.num_steps,
		)
