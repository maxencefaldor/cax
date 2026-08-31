"""Lenia module.

Use as a namespace, mirroring the classical genotype-to-phenotype pipeline:

	from breeder.cs import lenia

	genotype = lenia.sample(key, config)
	phenotype = lenia.develop(genotype, config)

"""

from .config import LeniaConfig, MutateConfig, SampleConfig
from .develop import SERIES, develop, observe, valid
from .genotype import Genotype
from .mutate import mutate
from .sample import SAMPLE_FNS, noise, sample, soliton, soliton_full

__all__ = [
	"SAMPLE_FNS",
	"SERIES",
	"Genotype",
	"LeniaConfig",
	"MutateConfig",
	"SampleConfig",
	"develop",
	"mutate",
	"noise",
	"observe",
	"sample",
	"soliton",
	"soliton_full",
	"valid",
]
