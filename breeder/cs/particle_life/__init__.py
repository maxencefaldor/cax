"""Particle Life module.

Use as a namespace, mirroring the classical genotype-to-phenotype pipeline:

	from breeder.cs import particle_life

	genotype = particle_life.sample(key, config)
	phenotype = particle_life.develop(genotype, config)

"""

from .config import MutateConfig, ParticleLifeConfig, SampleConfig
from .develop import SERIES, develop, valid
from .genotype import Genotype
from .mutate import mutate
from .sample import SAMPLE_FNS, blob, class_id, sample, uniform

__all__ = [
	"SAMPLE_FNS",
	"SERIES",
	"Genotype",
	"MutateConfig",
	"ParticleLifeConfig",
	"SampleConfig",
	"blob",
	"class_id",
	"develop",
	"mutate",
	"sample",
	"uniform",
	"valid",
]
