"""Generate one config file per experiment into breeder/configs/.

The defaults across the config models *are* the converged protocol (2026-08-31: 128²
world, single Aquarium soliton seed, tuned pose-invariant VAE, 1024 generations), so an
experiment states only its delta. Settled one-off experiments are removed once judged;
their definitions live in git history and their conclusions in notes/EXPERIMENTS.md.

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
	# --- The protocol itself, across seeds
	"base": {},
	"base_seed1": {"seed": 1},
	"base_seed2": {"seed": 2},
	"base_seed3": {"seed": 3},
	# --- Latent size axis (base is 8)
	"latent4": {"encoder": {"name": "vae", "latent_size": 4}},
	"latent16": {"encoder": {"name": "vae", "latent_size": 16}},
	# --- Hand-crafted metric descriptor: the ablation control for the learned one
	"metrics": {"encoder": None, "descriptor": METRIC_DESCRIPTOR},
	# --- Fixed texture descriptor. VGG activations are memory-heavy during evaluation:
	# minibatch_size 32 keeps the first conv layer at ~4 GB on-device instead of 32
	"vgg_descriptor": {
		"encoder": {"name": "vgg"},
		"descriptor": {"series": (("vgg", 1.0),)},
		"qd": {"minibatch_size": 32},
	},
	# --- Unsupervised homeostasis fitness (the paper's): minimize latent variance
	"homeostasis": {"fitness": {"name": "homeostasis"}},
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
	# --- Mutation-operator arms (2026-08-31): measured pathologies of the draft
	# operators, each varied alone against base. Dirichlet drift fixates weights
	# (6/8 rules dead by ~100 applications); the pseudo-count keeps rules resurrectable
	"weight_floor": {"complex_system": {"mutate": {"weight_floor": 0.5}}},
	# Fully derived operator: c=1600 gives weights the same ~1%-of-range steps as every
	# other parameter (at c=100 they take 27% relative steps, the largest in the
	# genotype); floor 0.5 is the measured knee (0.1 still leaves ~1 dead rule)
	"weight_tuned": {
		"complex_system": {"mutate": {"weight_concentration": 1600.0, "weight_floor": 0.5}}
	},
	# Does seed evolution earn anything? The seed is 49k of the genotype's ~49k+30 dims
	"state_frozen": {"complex_system": {"mutate": {"state_strategy": "frozen"}}},
	# Support-preserving seed mutation: no mass creation at the reflected 0 boundary
	"state_multiplicative": {"complex_system": {"mutate": {"state_strategy": "multiplicative"}}},
	# Seed replicates: one run per arm is one sample, and the arms sit within seed spread
	"weight_floor_seed1": {"seed": 1, "complex_system": {"mutate": {"weight_floor": 0.5}}},
	"state_frozen_seed1": {
		"seed": 1,
		"complex_system": {"mutate": {"state_strategy": "frozen"}},
	},
	"state_multiplicative_seed1": {
		"seed": 1,
		"complex_system": {"mutate": {"state_strategy": "multiplicative"}},
	},
	"weight_tuned_seed1": {
		"seed": 1,
		"complex_system": {"mutate": {"weight_concentration": 1600.0, "weight_floor": 0.5}},
	},
	# --- Pure random search: no mutation, fresh samples every generation
	"random_search": {
		"complex_system": {"sample": {"strategy": "noise"}},
		"qd": {"sample_ratio": 1.0},
	},
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
