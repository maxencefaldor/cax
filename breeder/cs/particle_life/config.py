"""Particle Life configuration."""

from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from breeder.core import ComplexSystem


class SampleConfig(BaseModel):
	"""Configuration of genotype sampling.

	Attributes:
		strategy: Sampling strategy (see `particle_life.sample`).
		blob_radius: Radius of the seeding disk, as a fraction of the torus (`blob`).

	"""

	model_config = ConfigDict(frozen=True)

	strategy: Literal["uniform", "blob"] = "blob"
	blob_radius: PositiveFloat = 0.1


class MutateConfig(BaseModel):
	"""Configuration of genotype mutation.

	Attributes:
		mutation_std: Mutation standard deviation, relative to each parameter's range.
		position_std: Mutation standard deviation of particle positions, as a fraction of
			the torus. Separate from `mutation_std` because positions are an arrangement
			of hundreds of particles rather than a handful of rule parameters: the same
			relative step would scramble the seed.

	"""

	model_config = ConfigDict(frozen=True)

	mutation_std: float = 0.01
	position_std: float = 0.002


class ParticleLifeConfig(BaseModel):
	"""Configuration of the Particle Life complex system and its genotype space.

	Particles of a few classes attract and repel each other by a distance-dependent force:
	always repelling closer than `beta · r_max`, then following the attraction matrix out
	to `r_max`. The evolved genotype is the attraction matrix, the force's crossover
	`beta`, and the initial arrangement.

	What is *not* evolved is deliberate. `r_max` sets the length unit that every metric is
	expressed in, so evolving it would make the population's measurements incomparable —
	the same reason Lenia freezes `R`. `dt`, `force_factor` and `velocity_half_life` are
	integration and time-scale parameters, phenotype-neutral in the continuum limit.

	Attributes:
		name: Discriminator tag of the complex system.
		num_particles: Number of particles. Pairwise forces make the cost quadratic in it.
		num_classes: Number of particle classes; the attraction matrix is square in it.
		num_steps: Number of simulation steps per development. The blob relaxes into its
			settled structure by roughly a thousand steps; measuring earlier measures the
			transient rather than the creature.
		num_frames: Number of final frames rendered. Rendering costs
			`resolution^2 * num_particles` per frame — on par with the simulation itself —
			so only the observed tail is drawn.
		resolution: Side length in pixels of the rendered frames.
		dt: Integration time step.
		force_factor: Global scaling of the interaction forces.
		velocity_half_life: Time over which friction halves a particle's velocity.
		r_max: Interaction radius, as a fraction of the torus. The system's length unit.
		beta_range: Range of the force's crossover radius, as a fraction of `r_max`.
		particle_radius: Rendered radius of a particle, as a fraction of the torus. At
			0.008 a particle is one pixel at resolution 128 and the creature renders as
			near-invisible specks; 0.016 gives it a legible body, for the encoder as much
			as for the eye.
		min_concentration: Minimum localization at every step for a phenotype to be valid.
		min_clustering: Minimum local density (1 = a uniform soup) at every step for a
			phenotype to be valid.
		sample: Genotype sampling configuration.
		mutate: Genotype mutation configuration.

	"""

	model_config = ConfigDict(frozen=True)

	name: Literal["particle_life"] = "particle_life"
	num_particles: PositiveInt = 256
	num_classes: PositiveInt = 4
	num_steps: PositiveInt = 1024
	num_frames: PositiveInt = 64
	resolution: PositiveInt = 128

	dt: PositiveFloat = 0.01
	force_factor: PositiveFloat = 1.0
	velocity_half_life: PositiveFloat = 0.01
	r_max: PositiveFloat = 0.15
	beta_range: tuple[float, float] = (0.1, 0.7)
	particle_radius: PositiveFloat = 0.016

	min_concentration: float = 0.5
	min_clustering: float = 2.0

	sample: SampleConfig = SampleConfig()
	mutate: MutateConfig = MutateConfig()

	def build(self) -> ComplexSystem:
		"""Build the Particle Life complex system bound to this config."""
		# Deferred import: the sibling modules import ParticleLifeConfig at load time
		from breeder.cs.particle_life import SERIES, develop, mutate, sample, valid

		return ComplexSystem(
			sample_fn=partial(sample, config=self),
			mutate_fn=partial(mutate, config=self),
			develop_fn=partial(develop, config=self),
			valid_fn=partial(valid, config=self),
			spatial_dims=(self.resolution, self.resolution),
			unit=self.r_max * self.resolution,
			series=SERIES,
			num_frames=self.num_frames,
		)
