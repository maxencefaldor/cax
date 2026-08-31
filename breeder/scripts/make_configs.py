"""Generate one config file per experiment into breeder/configs/.

The defaults across the config models *are* the converged protocol (2026-08-31: 128²
world, single Aquarium soliton seed, tuned pose-invariant VAE, 1024 generations), so an
experiment states only its delta.

Three rules keep the set small:

- **A config is a question, not a run.** Seeds and renames are command-line flags
  (`--config base.yaml --seed 1 --name base_seed1`), never files.
- **Per system, one config per descriptor family** — the learned one, the hand-crafted
  one, and where it exists the fixed pretrained one — plus the null control.
- **A wave is temporary.** The arms of a question under investigation live here while it
  is open and are deleted once judged; their definitions stay in git history and their
  conclusions in notes/EXPERIMENTS.md.

Usage: python -m breeder.scripts.make_configs
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path
from typing import Any

import yaml

from breeder.main import Config

METRIC_DESCRIPTOR = {"series": (("mass", 10.0), ("linear_velocity", 0.01))}
PARTICLE_DESCRIPTOR = {"series": (("radius", 1.0), ("clustering", 5.0))}

COMMON = {"complex_system": {"name": "lenia"}}

EXPERIMENTS = {
	# --- Lenia: the protocol, and one config per descriptor family
	"base": {},
	"metrics": {"encoder": None, "descriptor": METRIC_DESCRIPTOR},
	# VGG activations are memory-heavy during evaluation: minibatch_size 32 keeps the
	# first conv layer at ~4 GB on-device instead of 32
	"vgg_descriptor": {
		"encoder": {"name": "vgg"},
		"descriptor": {"series": (("vgg", 1.0),)},
		"qd": {"minibatch_size": 32},
	},
	# The null control: no mutation, fresh samples every generation. Without it no
	# result above means anything
	"random_search": {
		"complex_system": {"sample": {"strategy": "noise"}},
		"qd": {"sample_ratio": 1.0},
	},
	# --- Flow Lenia (arXiv:2212.07906): mass-conserving Lenia, seeded with the paper's
	# random patch. Mass is conserved, so localization is the gate that does the work.
	"flow_lenia": {"complex_system": {"name": "flow_lenia"}, "qd": {"num_init": 1024}},
	"flow_lenia_metrics": {
		"complex_system": {"name": "flow_lenia"},
		"encoder": None,
		"descriptor": {"series": (("concentration", 0.5), ("linear_velocity", 0.01))},
		"qd": {"num_init": 1024},
	},
	# --- Particle Life: the genotype is the attraction matrix, the force crossover and
	# the initial arrangement. Seeded from one disk, from which a soliton can persist.
	"particle_life": {"complex_system": {"name": "particle_life"}, "qd": {"num_init": 1024}},
	"particle_life_metrics": {
		"complex_system": {"name": "particle_life"},
		"encoder": None,
		"descriptor": PARTICLE_DESCRIPTOR,
		"qd": {"num_init": 1024},
	},
	# The prior's own diversity beat both evolved populations to the user's eye: keep
	# fresh prior samples flowing through the search
	"particle_life_sampled": {
		"complex_system": {"name": "particle_life"},
		"encoder": None,
		"descriptor": PARTICLE_DESCRIPTOR,
		"qd": {"num_init": 1024, "sample_ratio": 0.5},
	},
	# --- OPEN WAVE (2026-08-31): are the draft mutation operators pathological?
	# Measured: Dirichlet drift fixates 6 of 8 rules within ~100 applications, and the
	# seed's reflected Gaussian creates mass at the 0 boundary. Delete once judged.
	"weight_floor": {"complex_system": {"mutate": {"weight_floor": 0.5}}},
	"weight_tuned": {
		"complex_system": {"mutate": {"weight_concentration": 1600.0, "weight_floor": 0.5}}
	},
	"state_frozen": {"complex_system": {"mutate": {"state_strategy": "frozen"}}},
	"state_multiplicative": {"complex_system": {"mutate": {"state_strategy": "multiplicative"}}},
}


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
	"""Deep-merge overrides into base, dict fields recursively.

	Two sections are stated in full rather than refined, so that one variant's fields
	never leak into another's: `encoder` always, and `complex_system` whenever the
	override names a system — naming one *switches* system, omitting the name refines
	the base one.
	"""
	merged = dict(base)
	for key, value in overrides.items():
		switches_system = key == "complex_system" and "name" in value
		if (
			key != "encoder"
			and not switches_system
			and isinstance(value, dict)
			and isinstance(merged.get(key), dict)
		):
			merged[key] = merge(merged[key], value)
		else:
			merged[key] = value
	return merged


def main() -> None:
	"""Write every experiment's config file."""
	config_dir = Path(__file__).parent.parent / "configs"
	config_dir.mkdir(exist_ok=True)
	for path in config_dir.glob("*.yaml"):
		path.unlink()
	for name, overrides in EXPERIMENTS.items():
		config = Config.model_validate(merge(COMMON, {"name": name, **overrides}))
		(config_dir / f"{name}.yaml").write_text(
			yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
		)
		print(name)


if __name__ == "__main__":
	main()
