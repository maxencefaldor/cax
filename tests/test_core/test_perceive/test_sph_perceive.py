"""Tests for smoothed particle hydrodynamics perception."""

import jax
import jax.numpy as jnp
import pytest

from cax.core.perceive import Particles, SPHPerceive, poly6_kernel, spiky_gradient_kernel

SUPPORT_RADIUS = 0.1


@pytest.fixture
def cloud() -> Particles:
	"""Build a uniform cloud, dense enough for the estimators to mean something."""
	position = jax.random.uniform(jax.random.key(0), (4096, 2))
	return Particles(position=position, state=jnp.zeros((4096, 1)))


def interior(position: jax.Array) -> jax.Array:
	"""Mask particles away from the wrap, where a non-periodic comparison would differ."""
	return jnp.all((position > 0.25) & (position < 0.75), axis=-1)


def test_poly6_vanishes_outside_the_support() -> None:
	"""The kernel has compact support, and reaches zero smoothly rather than stepping to it."""
	distance = jnp.linspace(0.0, 2.0 * SUPPORT_RADIUS, 64)
	weight = poly6_kernel(jnp.square(distance), SUPPORT_RADIUS)

	assert jnp.all(weight[distance >= SUPPORT_RADIUS] == 0.0)
	assert jnp.all(weight[distance < SUPPORT_RADIUS] > 0.0)
	assert weight[0] == jnp.max(weight)


def test_spiky_gradient_points_inward_and_survives_the_origin() -> None:
	"""The gradient of a falling kernel points back towards the particle.

	Poly6's own gradient vanishes as two particles approach, leaving the closest neighbors
	contributing nothing to a gradient estimate. Spiky's does not, which is why it is the
	one used wherever a direction is wanted.
	"""
	displacement = jnp.array([[0.05, 0.0], [0.0, 0.02], [0.2, 0.0], [0.0, 0.0]])
	distance = jnp.linalg.norm(displacement, axis=-1)
	gradient = spiky_gradient_kernel(displacement, distance, SUPPORT_RADIUS)

	assert jnp.all(jnp.sum(gradient[:2] * displacement[:2], axis=-1) < 0.0)  # inward
	assert jnp.all(gradient[2] == 0.0)  # beyond the support
	assert jnp.all(jnp.isfinite(gradient[3]))  # coincident particles


def test_gradient_recovers_a_linear_field(cloud: Particles) -> None:
	"""The estimator should return the gradient of a field it is given."""
	slope = jnp.array([1.3, -0.7])
	particles = Particles(position=cloud.position, state=(cloud.position @ slope)[:, None])

	perception = SPHPerceive(support_radius=SUPPORT_RADIUS, log_normalize=False, period=1.0)(
		particles
	)
	gradient = perception[:, 2:4][interior(cloud.position)]

	relative_error = jnp.linalg.norm(jnp.mean(gradient, axis=0) - slope) / jnp.linalg.norm(slope)
	assert float(relative_error) < 0.1


def test_gradient_of_a_constant_field_vanishes(cloud: Particles) -> None:
	"""The difference form is what guarantees this, and it is why the estimator uses it.

	Summing the neighbors' states directly would leave a constant field with a non-zero
	gradient wherever the cloud is uneven, because the kernel gradients no longer cancel.
	"""
	particles = Particles(position=cloud.position, state=jnp.full((4096, 1), 2.5))

	perception = SPHPerceive(support_radius=SUPPORT_RADIUS, log_normalize=False, period=1.0)(
		particles
	)

	assert jnp.allclose(perception[:, 2:4], 0.0, atol=1e-4)


def test_average_matches_the_state_on_a_linear_field(cloud: Particles) -> None:
	"""A neighborhood average is the state itself where the field is linear.

	It is the *departure* from that, on a curved field, which stands in for a Laplacian.
	"""
	slope = jnp.array([1.3, -0.7])
	particles = Particles(position=cloud.position, state=(cloud.position @ slope)[:, None])

	perception = SPHPerceive(support_radius=SUPPORT_RADIUS, period=1.0)(particles)
	own, average = perception[:, :1], perception[:, 1:2]

	assert float(jnp.mean(jnp.abs(average - own)[interior(cloud.position)])) < 0.02


def test_density_gradient_vanishes_on_a_uniform_cloud(cloud: Particles) -> None:
	"""No direction is denser than another, so the cue that says otherwise should be quiet."""
	perception = SPHPerceive(support_radius=SUPPORT_RADIUS, log_normalize=False, period=1.0)(cloud)
	single_term = 30.0 / (jnp.pi * SUPPORT_RADIUS**5)

	density_gradient = perception[:, 4:6][interior(cloud.position)]

	assert float(jnp.mean(jnp.abs(density_gradient)) / single_term) < 0.05


def test_perception_size(cloud: Particles) -> None:
	"""The advertised width is the width produced."""
	particles = Particles(position=cloud.position, state=jnp.zeros((4096, 7)))

	perception = SPHPerceive(support_radius=SUPPORT_RADIUS, period=1.0)(particles)

	assert perception.shape == (
		4096,
		SPHPerceive.perception_size(channel_size=7, num_spatial_dims=2),
	)


def test_perception_is_differentiable(cloud: Particles) -> None:
	"""Positions are what the rule moves, so the gradient has to reach them."""
	particles = Particles(position=cloud.position[:256], state=cloud.state[:256])
	perceive = SPHPerceive(support_radius=SUPPORT_RADIUS, period=1.0)

	gradient = jax.grad(lambda p: jnp.sum(jnp.square(perceive(p))))(particles)

	assert gradient.position.shape == particles.position.shape
	assert jnp.all(jnp.isfinite(gradient.position))
	assert jnp.any(gradient.position != 0.0)


def test_perception_differentiates_where_nothing_is_happening(cloud: Particles) -> None:
	"""A particle whose neighborhood is uniformly zero still has to produce a gradient.

	The compression applied to the two gradients divides by a length, and a length is a
	square root whose derivative is unbounded at the origin. Most particles start in a
	neighborhood where every state is zero, so this is the common case rather than an edge
	one, and getting it wrong turns the whole model to `nan` on the first step.
	"""
	particles = Particles(position=cloud.position[:512], state=jnp.zeros((512, 8)))
	perceive = SPHPerceive(support_radius=SUPPORT_RADIUS, mass=1.0 / 512, period=1.0)

	gradient = jax.grad(lambda p: jnp.sum(jnp.square(perceive(p))))(particles)

	assert jnp.all(jnp.isfinite(gradient.position))
	assert jnp.all(jnp.isfinite(gradient.state))


def test_log_normalize_keeps_direction_and_compresses_length(cloud: Particles) -> None:
	"""Compression is what makes the perception safe to hand to a network.

	A kernel gradient carries the support radius to a negative power, so it arrives orders
	of magnitude larger than the states beside it. The direction is the part that means
	something; the magnitude only has to be on a comparable scale.
	"""
	slope = jnp.array([1.3, -0.7])
	particles = Particles(position=cloud.position, state=(cloud.position @ slope)[:, None])

	raw = SPHPerceive(support_radius=SUPPORT_RADIUS, log_normalize=False, period=1.0)(particles)
	compressed = SPHPerceive(support_radius=SUPPORT_RADIUS, period=1.0)(particles)

	inside = interior(cloud.position)
	raw_gradient, compressed_gradient = raw[:, 2:4][inside], compressed[:, 2:4][inside]

	raw_length = jnp.linalg.norm(raw_gradient, axis=-1)
	compressed_length = jnp.linalg.norm(compressed_gradient, axis=-1)
	assert jnp.allclose(compressed_length, jnp.log1p(raw_length), atol=1e-4)

	# The direction survives.
	cosine = jnp.sum(raw_gradient * compressed_gradient, axis=-1) / (
		raw_length * compressed_length + 1e-9
	)
	assert float(jnp.min(cosine)) > 0.999


gpu_only = pytest.mark.skipif(
	jax.default_backend() != "gpu", reason="the fused kernels are written for GPU"
)


def both_ways(num_particles: int, *, fused: bool) -> SPHPerceive:
	"""Build the same perception either way, so the two can be compared."""
	return SPHPerceive(support_radius=0.2, mass=1.0 / num_particles, period=1.0, fused=fused)


@gpu_only
def test_fused_matches_the_array_path() -> None:
	"""The kernels are only worth having if they compute what the array version computes.

	Everything downstream rests on this: a perception that is close but not equal trains a
	different model, quietly.
	"""
	num_particles, channel_size = 1024, 16
	particles = Particles(
		position=jax.random.uniform(jax.random.key(1), (num_particles, 2)),
		state=jax.random.normal(jax.random.key(2), (num_particles, channel_size)) * 0.3,
	)

	array = both_ways(num_particles, fused=False)(particles)
	fused = both_ways(num_particles, fused=True)(particles)

	assert fused.shape == array.shape
	assert jnp.max(jnp.abs(fused - array)) / jnp.max(jnp.abs(array)) < 1e-3


@gpu_only
def test_fused_gradients_match_the_array_path() -> None:
	"""A kernel is opaque to autodiff, so its derivative is written by hand.

	A wrong derivative does not raise; it trains something else. This is the check that the
	hand-derived backward is the derivative of the forward beside it.
	"""
	num_particles, channel_size = 1024, 16
	particles = Particles(
		position=jax.random.uniform(jax.random.key(1), (num_particles, 2)),
		state=jax.random.normal(jax.random.key(2), (num_particles, channel_size)) * 0.3,
	)

	def loss(particles: Particles, fused: bool) -> jax.Array:
		return jnp.sum(jnp.square(both_ways(num_particles, fused=fused)(particles)))

	array = jax.grad(lambda p: loss(p, False))(particles)
	fused = jax.grad(lambda p: loss(p, True))(particles)

	for ours, theirs in ((fused.position, array.position), (fused.state, array.state)):
		assert jnp.max(jnp.abs(ours - theirs)) / jnp.max(jnp.abs(theirs)) < 1e-3


@gpu_only
def test_fused_maps_over_a_batch() -> None:
	"""The kernel takes one cloud, but training perceives a batch of them."""
	num_particles, channel_size = 1024, 8
	batch = Particles(
		position=jax.random.uniform(jax.random.key(1), (3, num_particles, 2)),
		state=jax.random.normal(jax.random.key(2), (3, num_particles, channel_size)) * 0.3,
	)

	array = both_ways(num_particles, fused=False)(batch)
	fused = both_ways(num_particles, fused=True)(batch)

	assert fused.shape == array.shape
	assert jnp.max(jnp.abs(fused - array)) / jnp.max(jnp.abs(array)) < 1e-3


@gpu_only
@pytest.mark.parametrize("num_particles", [64, 100, 1000, 1024, 1237, 4096, 5000])
def test_fused_matches_at_any_cloud_size(num_particles: int) -> None:
	"""The tiling is static, but the cloud need not be a multiple of it.

	A count the tiles do not divide is padded out and the extra entries are masked away, so
	sizes that are awkward for the kernel must still give the array version's answer rather
	than a slightly wrong one.
	"""
	channel_size = 8
	particles = Particles(
		position=jax.random.uniform(jax.random.key(1), (num_particles, 2)),
		state=jax.random.normal(jax.random.key(2), (num_particles, channel_size)) * 0.3,
	)

	array = both_ways(num_particles, fused=False)(particles)
	fused = both_ways(num_particles, fused=True)(particles)

	assert fused.shape == array.shape
	assert jnp.max(jnp.abs(fused - array)) / jnp.max(jnp.abs(array)) < 1e-3
