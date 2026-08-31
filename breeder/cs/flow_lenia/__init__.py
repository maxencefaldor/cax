"""Flow Lenia module.

Flow Lenia (arXiv:2212.07906) is Lenia with a mass-conserving update. The genotype
space is Lenia's, so `Genotype`, `sample`, `mutate` and `valid` are Lenia's own — only
the config and the development differ:

	from breeder.cs import flow_lenia

	genotype = flow_lenia.sample(key, config)
	phenotype = flow_lenia.develop(genotype, config)

"""

from ..lenia import SERIES, Genotype, mutate, sample, valid
from .config import FlowLeniaConfig
from .develop import develop

__all__ = [
	"SERIES",
	"FlowLeniaConfig",
	"Genotype",
	"develop",
	"mutate",
	"sample",
	"valid",
]
