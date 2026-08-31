"""Population report module.

Generates a self-contained explorer for a finished run, written to
`<run_dir>/report/index.html`:

- Individuals: every individual as a looping, uncentered video thumbnail (raw view, so
  motion is visible), with client-side sorting and filtering.
- Map: the population in the fixed reference space (see `core.diversity`), with live
  video previews on hover.
- Progress: metric curves over generations, from the run's `log.csv`.
- Config: the experiment's full YAML config.
"""

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import yaml

from breeder.core import diversity
from breeder.core.dns import DNSState

_ASSETS = Path(__file__).parent


def _page(name: str, report: dict[str, object]) -> str:
	"""Assemble the self-contained report page from the package's assets."""
	return (
		(_ASSETS / "template.html")
		.read_text()
		.replace("__STYLE__", (_ASSETS / "style.css").read_text())
		.replace("__SCRIPT__", (_ASSETS / "script.js").read_text())
		.replace("__NAME__", name)
		.replace("__REPORT__", json.dumps(report))
	)


def _progress(run_dir: Path, config) -> dict[str, list[float]]:
	"""Read the run's metric curves from log.csv, with evaluation counts as x axis."""
	with open(run_dir / "log.csv", newline="") as file:
		rows = list(csv.DictReader(file))
	names = [
		"generation",
		"num_valid",
		"best_fitness",
		"mean_fitness",
		"diversity",
		"vendi",
		"vendi_cos",
		"child_valid",
		"variance",
	]
	progress = {
		name: [float(row[name]) if row.get(name) else float("nan") for row in rows]
		for name in names
	}
	progress["evaluations"] = [
		config.qd.num_init + generation * config.qd.num_children
		for generation in progress["generation"]
	]
	return progress


def _labels(config) -> tuple[str, str]:
	"""Human-readable fitness and descriptor definitions."""
	if getattr(config.fitness, "name", None) == "homeostasis":
		fitness = f"homeostasis({config.fitness.series}) · window {config.fitness.window}"
	else:
		sign = "-" if config.fitness.sign == -1 else ""
		channel = f"[{config.fitness.channel}]" if config.fitness.channel is not None else ""
		fitness = (
			f"{sign}{config.fitness.reduce} {config.fitness.series}{channel}"
			f" · window {config.fitness.window}"
		)
	encoder = config.encoder
	labels = {encoder.series: f"{encoder.series} ({encoder.name})"} if encoder.series else {}
	descriptor = (
		"mean "
		+ ", ".join(labels.get(name, name) for name, _ in config.descriptor.series)
		+ f" · window {config.descriptor.window}"
	)
	return fitness, descriptor


def write_report(state: DNSState, run_dir: Path, config, *, minibatch_size: int) -> None:
	"""Write the population explorer for a run.

	Args:
		state: Final DNS state.
		run_dir: Run directory; the report is written to `run_dir/report/`.
		config: The run's experiment config.
		minibatch_size: Individuals developed per sequential minibatch.

	"""
	report_dir = run_dir / "report"
	video_dir = report_dir / "videos"
	video_dir.mkdir(parents=True, exist_ok=True)
	thumb_dir = report_dir / "thumbs"
	thumb_dir.mkdir(parents=True, exist_ok=True)

	# Develop the population, fittest first, with the raw (uncentered) view
	order = jnp.argsort(state.fitness, descending=True)
	num = min(config.num_thumbnails, order.shape[0])
	population = jax.tree.map(lambda x: x[order[:num]], state.population)
	fitness = state.fitness[order[:num]]
	valid = fitness != -jnp.inf

	cs = config.complex_system.build()

	@jax.jit
	def develop_population(population):
		return jax.lax.map(
			lambda genotype: cs.develop_fn(genotype, center=False),
			population,
			batch_size=minibatch_size,
		)

	phenotypes = develop_population(population)

	# 2D map: PCA of reference features of the final centered observation
	reference_fn = diversity.reference_encoder()
	last_observations = state.observations[order[:num], -1]
	features = diversity.reference_features(reference_fn, last_observations, unit=cs.unit)
	mean_distance = diversity.mean_pairwise_distance(features, valid)

	centered = features - jnp.mean(features, axis=0)
	_, _, vt = jnp.linalg.svd(centered, full_matrices=False)
	coords = centered @ vt[:2].T
	low, high = jnp.min(coords, axis=0), jnp.max(coords, axis=0)
	coords = (coords - low) / jnp.maximum(high - low, 1e-8)

	# Windowed mean of every scalar series the system emits, for card stats
	window = config.fitness.window
	series_values = {
		name: jnp.mean(values[:, -window:], axis=1)
		for name, values in phenotypes.series.items()
		if values.ndim == 2
	}

	fitness_label, descriptor_label = _labels(config)
	report = {
		"summary": {
			"fitness": fitness_label,
			"descriptor": descriptor_label,
			"num_valid": int(jnp.sum(valid)),
			"best_fitness": float(state.best_fitness),
			"diversity": float(mean_distance),
			"vendi": float(diversity.vendi_score(features, valid)),
			"vendi_cos": float(diversity.vendi_score(features, valid, correlation=True)),
		},
		"config_yaml": yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
		"progress": _progress(run_dir, config),
		"population": [
			{
				"id": rank,
				"fitness": float(fitness[rank]) if bool(valid[rank]) else None,
				"metrics": {name: float(values[rank]) for name, values in series_values.items()},
				"x": float(coords[rank, 0]),
				"y": float(coords[rank, 1]),
				"rank01": rank / max(num - 1, 1),
			}
			for rank in range(num)
		],
	}

	# Encode only missing videos and posters so regenerating a report is fast. The
	# poster is the final frame — the creature as selection judged it. Cards show
	# posters as plain lazy-loaded images and overlay the video only once it has
	# decoded, so the grid never depends on the browser's video-decoder budget.
	frames = None
	for rank in range(num):
		video_path = video_dir / f"{rank}.mp4"
		thumb_path = thumb_dir / f"{rank}.jpg"
		if video_path.exists() and thumb_path.exists():
			continue
		if frames is None:
			frames = np.asarray(phenotypes.frames)
		if not video_path.exists():
			mediapy.write_video(video_path, frames[rank], fps=16)
		if not thumb_path.exists():
			mediapy.write_image(thumb_path, frames[rank, -1])

	html = _page(config.name, report)
	(report_dir / "index.html").write_text(html)
	print(f"Report: {report_dir / 'index.html'}")
