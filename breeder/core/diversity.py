"""Diversity evaluation module.

Measuring diversity with the searched descriptor is circular: a VAE descriptor is
retrained during the search, so its distances are comparable neither across generations
nor across experiments. Diversity is therefore evaluated in a fixed *reference space*:
frames are cropped to a fixed physical window, resized to a canonical resolution, and
embedded by frozen pretrained VGG16 features, identical for every experiment.

The reference space used to be a frozen *random* convolutional encoder, on the
Johnson-Lindenstrauss argument that a random projection approximately preserves
distances. It does -- but preserving pixel distance is not the goal. A population of
near-identical creatures differing only in phase and micro-detail is genuinely far apart
in pixel space while being visually identical, so the random space rated a collapsed
population as diverse. Measured against the eye's verdict on the roll/no-roll pair, every
VGG measure agreed on both seeds and the random ones were a coin flip; see
`notes/EXPERIMENTS.md`. Perceptual features collapse exactly the nuisance directions,
which is also why VGG activations underpin LPIPS.

Pixel-space phenotype variance (the official Leniabreeder measure) complements it.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .vgg import VGG16

REFERENCE_SPATIAL_DIMS = (64, 64)
REFERENCE_LAYER = 11  # third block's last relu, 256 channels
REFERENCE_CROP_RADII = 4.0
REFERENCE_BATCH_SIZE = 128


def center_crop(frames: Array, size: int) -> Array:
	"""Crop the center `size`x`size` window, zero-padding if the frames are smaller.

	Frames must already be centered on the pattern. Cropping to a fixed *physical* window
	(`REFERENCE_CROP_RADII` kernel radii) before embedding makes the reference space
	comparable across world sizes: without it, the same creature in a larger world covers
	a smaller fraction of the frame and all distances shrink.

	Args:
		frames: RGB frames with shape `(N, H, W, 3)`.
		size: Side length of the crop, in pixels.

	Returns:
		RGB frames with shape `(N, size, size, 3)`.

	"""
	height, width = frames.shape[-3], frames.shape[-2]
	pad_height = max(size - height, 0)
	pad_width = max(size - width, 0)
	frames = jnp.pad(
		frames,
		(
			(0, 0),
			(pad_height // 2, pad_height - pad_height // 2),
			(pad_width // 2, pad_width - pad_width // 2),
			(0, 0),
		),
	)
	start_height = (frames.shape[-3] - size) // 2
	start_width = (frames.shape[-2] - size) // 2
	return frames[..., start_height : start_height + size, start_width : start_width + size, :]


class ReferenceEncoder(nnx.Module):
	"""Frozen pretrained encoder defining the reference space.

	VGG16's ImageNet feature stack up to `REFERENCE_LAYER`, spatially averaged. Frames are
	resized to `REFERENCE_SPATIAL_DIMS` first: the crop preceding this embedding is a
	fixed *physical* window, so its pixel size varies with world resolution and with the
	system, and the resize is what makes one reference space serve all of them.
	"""

	def __init__(self, *, layer: int = REFERENCE_LAYER, rngs: nnx.Rngs):
		"""Initialize the reference encoder.

		Args:
			layer: Index into VGG16's feature stack at which activations are taken.
			rngs: rng key (placeholder -- weights are overwritten with VGG16's).

		"""
		self.vgg = VGG16(layer=layer, rngs=rngs)

	def __call__(self, frames: Array) -> Array:
		"""Embed frames into the reference space.

		Args:
			frames: RGB frames with dtype uint8 and shape `(N, H, W, 3)`.

		Returns:
			Reference features with shape `(N, 256)`.

		"""
		x = frames.astype(jnp.float32)
		x = jax.image.resize(x, (x.shape[0], *REFERENCE_SPATIAL_DIMS, x.shape[-1]), method="linear")
		return jnp.mean(self.vgg(x), axis=(-3, -2))


def reference_encoder() -> ReferenceEncoder:
	"""Create the canonical reference encoder, identical across all experiments."""
	return ReferenceEncoder(rngs=nnx.Rngs(0))


def mean_pairwise_distance(features: Array, valid: Array) -> Array:
	"""Mean pairwise Euclidean distance among valid individuals.

	Args:
		features: Array with shape `(N, D)`.
		valid: Boolean array with shape `(N,)`.

	Returns:
		Scalar mean distance over valid pairs (nan if fewer than two valid).

	"""
	pair = valid[:, None] & valid[None, :]
	pair = jnp.fill_diagonal(pair, False, inplace=False)
	distance = jnp.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
	return jnp.mean(distance, where=pair)


def vendi_score(features: Array, valid: Array, *, correlation: bool = False) -> Array:
	"""Effective number of distinct individuals (Vendi score) in feature space.

	The Vendi score [1] is the exponential of the Shannon entropy of the eigenvalues of
	the normalized similarity matrix: it behaves like an "effective species count" —
	`N` identical individuals score 1, `N` mutually dissimilar individuals score `N`.

	Computed here with a *covariance* kernel: features are centered on the population
	mean and the kernel is scaled to unit trace. Centering matters because raw ReLU
	features are non-negative, so plain cosine similarities crowd near 1 and compress the
	score toward 1. Not rescaling each individual to unit length matters just as much: a
	correlation kernel discards how far an individual sits from the mean, and in a
	collapsed population the residuals are near-pure noise, which in high dimension is
	near-orthogonal — so collapse scored close to its maximum. See `notes/EXPERIMENTS.md`.

	References:
		[1] The Vendi Score: A Diversity Evaluation Metric for Machine Learning,
			Friedman and Dieng, 2023.

	Args:
		features: Array with shape `(N, D)`.
		valid: Boolean array with shape `(N,)`.
		correlation: Whether to rescale each centered feature to unit length, giving the
			correlation kernel instead. Reported alongside as `vendi_cos` — it is the
			measure that disagreed with the eye, kept so the disagreement stays visible.

	Returns:
		Scalar Vendi score in `[1, num_valid]` (0 if none valid).

	"""
	num_valid = jnp.sum(valid)
	mean = jnp.mean(features, axis=0, where=valid[:, None])
	x = jnp.where(valid[:, None], features - mean, 0.0)
	if correlation:
		x = jnp.where(valid[:, None], x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8), 0.0)
	else:
		x = x / (jnp.sqrt(jnp.sum(jnp.square(x)) / jnp.maximum(num_valid, 1)) + 1e-8)

	kernel = x @ x.T
	eigenvalues = jnp.linalg.eigvalsh(kernel / jnp.maximum(num_valid, 1))
	p = jnp.clip(eigenvalues, 1e-12, None)
	entropy = -jnp.sum(jnp.where(eigenvalues > 1e-12, p * jnp.log(p), 0.0))
	return jnp.where(num_valid > 0, jnp.exp(entropy), 0.0)


def phenotype_variance(frames: Array, valid: Array) -> Array:
	"""Mean per-pixel variance across valid individuals (official Leniabreeder measure).

	Args:
		frames: RGB frames with dtype uint8 and shape `(N, H, W, 3)`.
		valid: Boolean array with shape `(N,)`.

	Returns:
		Scalar mean variance over pixels.

	"""
	x = frames.astype(jnp.float32) / 255.0
	variance = jnp.var(x, axis=0, where=valid[:, None, None, None])
	return jnp.mean(variance)


def reference_features(reference_fn: nnx.Module, frames: Array, *, unit: float) -> Array:
	"""Embed centered frames into the reference space: fixed physical crop, then encode.

	The crop side is `REFERENCE_CROP_RADII` physical units, so the window is comparable
	across world sizes and systems. Embedding runs in fixed-size chunks: VGG16's
	activations at `REFERENCE_LAYER` are two orders of magnitude larger than the frames
	that produce them, so one call over a 1024-individual population would peak at
	several GiB — chunking bounds that at one batch, and the features themselves are
	small.

	Args:
		reference_fn: Frozen reference encoder.
		frames: Centered RGB frames with dtype uint8 and shape `(N, H, W, 3)`.
		unit: Physical unit of the system, in pixels (`ComplexSystem.unit`).

	Returns:
		Reference features with shape `(N, feature_size)`.

	"""
	frames = center_crop(frames, round(REFERENCE_CROP_RADII * unit))
	batches = range(0, frames.shape[0], REFERENCE_BATCH_SIZE)
	return jnp.concatenate([reference_fn(frames[i : i + REFERENCE_BATCH_SIZE]) for i in batches])


def population_metrics(
	fitness: Array, frames: Array, reference_fn: nnx.Module, *, unit: float
) -> dict[str, Array]:
	"""Compute the search-independent metric battery of an evaluated population.

	Args:
		fitness: Fitness with shape `(N,)`; invalid individuals are `-inf`.
		frames: Final observation frames with shape `(N, H, W, 3)`, centered.
		reference_fn: Frozen reference encoder (see module docstring).
		unit: Physical unit of the system, in pixels (`ComplexSystem.unit`).

	Returns:
		Named scalar metrics: validity count, fitness statistics, and the diversity
		battery (reference-space spread, both Vendi kernels, pixel variance).

	"""
	valid = fitness != -jnp.inf
	features = reference_features(reference_fn, frames, unit=unit)
	return {
		"num_valid": jnp.sum(valid),
		"max_fitness": jnp.max(fitness, initial=-jnp.inf, where=valid),
		"mean_fitness": jnp.mean(fitness, where=valid),
		"diversity": mean_pairwise_distance(features, valid),
		"vendi": vendi_score(features, valid),
		"vendi_cos": vendi_score(features, valid, correlation=True),
		"variance": phenotype_variance(frames, valid),
	}
