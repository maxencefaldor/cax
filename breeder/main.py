"""Run a breeder experiment: Dominated Novelty Search over a complex system.

One experiment is one config. All fields are settable from the command line, e.g.:

	python -m breeder.main --name dev --fitness.series linear_velocity
	python -m breeder.main --complex-system.sample.strategy soliton --qd.num-generations 1024

A run directory stores the config (`config.yaml`), metrics (`log.csv`), asynchronous
orbax checkpoints of the full evolution state, and the population explorer (`report/`).

Evaluation is sharded over all visible devices (mesh + sharding constraint + jit); use
`CUDA_VISIBLE_DEVICES` to pin a run to specific GPUs.
"""

import csv
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Self

import jax
import jax.numpy as jnp
import yaml
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from pydantic import (
	BaseModel,
	BeforeValidator,
	ConfigDict,
	Field,
	NonNegativeInt,
	PositiveInt,
	model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from breeder.core import (
	DescriptorConfig,
	EncoderConfig,
	FitnessConfig,
	QDConfig,
	ReductionFitnessConfig,
	VAEEncoderConfig,
	checkpoint,
	diversity,
)
from breeder.cs.flow_lenia import FlowLeniaConfig
from breeder.cs.lenia import LeniaConfig
from breeder.cs.particle_life import ParticleLifeConfig
from breeder.report import write_report

# The complex systems an experiment can search. Adding one is a module plus this line
ComplexSystemConfig = Annotated[
	LeniaConfig | FlowLeniaConfig | ParticleLifeConfig, Field(discriminator="name")
]

# YAML `encoder: null` means the no-op encoder
NullableEncoderConfig = Annotated[
	EncoderConfig, BeforeValidator(lambda value: {"name": "none"} if value is None else value)
]


class Config(BaseModel):
	"""Experiment configuration: one experiment is one config.

	A pure model: validating a config never consults the command line or environment
	(only `CliConfig`, the command-line adapter, does).
	"""

	model_config = ConfigDict(frozen=True)

	name: str = "dev"
	seed: int = 0

	complex_system: ComplexSystemConfig = LeniaConfig()
	qd: QDConfig = QDConfig()
	encoder: NullableEncoderConfig = VAEEncoderConfig()
	fitness: FitnessConfig = ReductionFitnessConfig()
	descriptor: DescriptorConfig = DescriptorConfig()

	train_interval: PositiveInt = 32
	bootstrap: bool = True
	log_interval: PositiveInt = 8
	checkpoint_interval: PositiveInt = 32
	num_thumbnails: NonNegativeInt = 1024
	output_dir: str = "breeder/output"

	@property
	def reencode(self) -> bool:
		"""Whether descriptors must be re-encoded after a refit."""
		return self.encoder.trainable and any(
			series == self.encoder.series for series, _ in self.descriptor.series
		)

	@model_validator(mode="after")
	def check_divisibility(self) -> Self:
		"""Enforce the divisibility relations the jitted pipeline relies on."""
		device_count = jax.device_count()
		if self.qd.minibatch_size % device_count:
			raise ValueError(
				f"minibatch_size {self.qd.minibatch_size} must be a multiple of the "
				f"device count {device_count}"
			)
		if self.qd.num_children % self.qd.minibatch_size:
			raise ValueError("num_children must be a multiple of minibatch_size")
		if self.qd.population_size % self.qd.minibatch_size:
			raise ValueError("population_size must be a multiple of minibatch_size")
		if self.qd.num_generations % self.train_interval:
			raise ValueError("num_generations must be a multiple of train_interval")
		if self.checkpoint_interval % self.train_interval:
			raise ValueError("checkpoint_interval must be a multiple of train_interval")
		return self

	@model_validator(mode="after")
	def check_series(self) -> Self:
		"""Fail at parse time on series mistakes that would otherwise be silent or late.

		An unknown series name would only KeyError deep inside the jitted evaluate; a
		window larger than the encoder's would silently clamp (`[-window:]`); and a
		descriptor mixing the trainable series with system series would crash at the
		first refit, `train_interval` generations in.
		"""
		cs = self.complex_system.build()
		if self.encoder.window > cs.num_frames:
			raise ValueError(
				f"encoder.window {self.encoder.window} exceeds the {cs.num_frames} frames "
				f"{self.complex_system.name} renders"
			)

		available = cs.series
		if self.encoder.series is not None:
			available = (*available, self.encoder.series)

		references = (
			("fitness", self.fitness.series, self.fitness.window),
			*(("descriptor", s, self.descriptor.window) for s, _ in self.descriptor.series),
		)
		for consumer, series, window in references:
			if series not in available:
				raise ValueError(f"{consumer} series {series!r} is not in {available}")
			if series == self.encoder.series and window > self.encoder.window:
				raise ValueError(
					f"{consumer}.window {window} exceeds the encoder window {self.encoder.window}"
				)

		descriptor_series = {series for series, _ in self.descriptor.series}
		if self.reencode and len(descriptor_series) > 1:
			raise ValueError(
				"a descriptor mixing the trainable series with system series cannot "
				"be re-encoded after a refit"
			)
		return self


class CliConfig(Config, BaseSettings):
	"""Command-line adapter: every `Config` field as a flag, plus the run controls.

	Only this class reads the command line; a `Config` loaded from a file stays exactly
	what the file says.
	"""

	model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True, frozen=True)

	config: str = Field(default="", exclude=True)


def main(config: Config) -> None:
	"""Run the experiment."""
	run_dir = Path(config.output_dir) / f"{config.name}_{datetime.now():%Y-%m-%d_%H%M%S}"
	run_dir.mkdir(parents=True)
	(run_dir / "config.yaml").write_text(
		yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
	)
	print(f"Run directory: {run_dir}")

	# Mesh over all visible devices; evaluation is sharded over the child_individuals axis
	mesh = jax.make_mesh(
		(jax.device_count(),), ("device",), axis_types=(jax.sharding.AxisType.Auto,)
	)
	data_sharding = NamedSharding(mesh, PartitionSpec("device"))
	replicated_sharding = NamedSharding(mesh, PartitionSpec())

	key = jax.random.key(config.seed)

	cs = config.complex_system.build()
	dns = config.qd.build(cs.sample_fn, cs.mutate_fn)
	fitness_fn = config.fitness.build()
	descriptor_fn = config.descriptor.build()
	reference_fn = jax.device_put(diversity.reference_encoder(), replicated_sharding)

	start_time = time.monotonic()

	def log_row(metrics) -> None:
		"""Host callback: append one generation's metrics to `log.csv`, print periodically."""
		row = {name: value.item() for name, value in metrics.items()}
		row["time"] = time.monotonic() - start_time
		path = run_dir / "log.csv"
		write_header = not path.exists()
		with open(path, "a", newline="") as file:
			writer = csv.DictWriter(file, fieldnames=list(row))
			if write_header:
				writer.writeheader()
			writer.writerow(row)
		if row["generation"] % config.log_interval == 0:
			evaluations = config.qd.num_init + row["generation"] * config.qd.num_children
			print(
				f"generation {row['generation']:4d} | valid {row['num_valid']:4d} "
				f"| best {row['best_fitness']:.4f} | mean {row['mean_fitness']:.4f} "
				f"| diversity {row['diversity']:.4f} | {row['time']:.0f}s "
				f"| {evaluations / row['time']:.0f} evals/s"
			)

	key, subkey = jax.random.split(key)
	encoder_fn = jax.device_put(config.encoder.build(subkey, cs), replicated_sharding)

	def evaluate_one(genotype, encoder_fn):
		phenotype = cs.develop_fn(genotype)
		observation = phenotype.frames[-config.encoder.window :]
		phenotype = replace(phenotype, series=phenotype.series | encoder_fn.encode(observation))
		fitness = fitness_fn(phenotype)
		descriptor = descriptor_fn(phenotype)
		invalid = ~cs.valid_fn(phenotype) | jnp.isnan(fitness) | jnp.any(jnp.isnan(descriptor))
		fitness = jnp.where(invalid, -jnp.inf, fitness)
		return fitness, descriptor, observation

	def evaluate(genotypes, encoder_fn):
		return jax.lax.map(
			lambda genotype: evaluate_one(genotype, encoder_fn),
			genotypes,
			batch_size=config.qd.minibatch_size,
		)

	evaluate_jit = nnx.jit(evaluate)

	@nnx.jit
	def encode_population(observations, encoder_fn):
		"""Recompute the population's descriptors from the archived observations."""

		def encode_one(observation):
			return descriptor_fn.reduce(encoder_fn.encode(observation))

		return jax.lax.map(encode_one, observations, batch_size=config.qd.minibatch_size)

	@nnx.jit
	def run_block(key, state, generation, encoder_fn):
		"""Advance the search by train_interval generations in one compiled scan.

		Each generation logs its metrics through the `log_row` host callback.
		"""

		def scan_step(carry, key):
			state, generation = carry
			child_individuals = jax.lax.with_sharding_constraint(dns.ask(key, state), data_sharding)
			fitness, descriptor, observations = evaluate(child_individuals, encoder_fn)
			state = dns.tell(state, child_individuals, fitness, descriptor, observations)
			generation = generation + 1

			metrics = {
				"generation": generation,
				"best_fitness": state.best_fitness,
				"child_valid": jnp.mean(fitness != -jnp.inf),
			} | diversity.population_metrics(
				state.fitness, state.observations[:, -1], reference_fn, unit=cs.unit
			)
			jax.debug.callback(log_row, metrics)
			return (state, generation), None

		keys = jax.random.split(key, config.train_interval)
		(state, _), _ = jax.lax.scan(scan_step, (state, generation), keys)
		return state

	# Initialize the evolution state
	manager = checkpoint.checkpoint_manager(run_dir / "checkpoints")
	print("Initializing population...")
	key, key_init, key_fit = jax.random.split(key, 3)
	population = jax.vmap(dns.sample_fn)(jax.random.split(key_init, config.qd.num_init))
	fitness, descriptor, observations = evaluate_jit(population, encoder_fn)

	if config.bootstrap:
		encoder_fn = jax.device_put(
			encoder_fn.refit(key_fit, observations, valid=fitness != -jnp.inf), replicated_sharding
		)
		if config.reencode:
			descriptor = encode_population(observations, encoder_fn)
	state = jax.device_put(
		dns.init(population, fitness, descriptor, observations), replicated_sharding
	)

	for block_start in range(0, config.qd.num_generations, config.train_interval):
		key, subkey = jax.random.split(key)
		state = run_block(subkey, state, block_start, encoder_fn)
		generation = block_start + config.train_interval

		if generation < config.qd.num_generations:
			key, subkey = jax.random.split(key)
			encoder_fn = jax.device_put(
				encoder_fn.refit(subkey, state.observations, valid=state.fitness != -jnp.inf),
				replicated_sharding,
			)
			if config.reencode:
				state = replace(state, descriptor=encode_population(state.observations, encoder_fn))

		if generation % config.checkpoint_interval == 0:
			checkpoint.save(manager, generation, state, encoder_fn, key)

	manager.wait_until_finished()

	# Population explorer
	print("Writing report...")
	write_report(state, run_dir, config, minibatch_size=config.qd.minibatch_size)
	print("Done.")


def load_config(path: Path) -> Config:
	"""Load an experiment config from a YAML file."""
	return Config.model_validate(yaml.safe_load(path.read_text()))


if __name__ == "__main__":
	cli = CliConfig()
	main(load_config(Path(cli.config)) if cli.config else cli)
