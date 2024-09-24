"""Tests for the VAE module."""

import jax
import jax.numpy as jnp
import pytest
from cax.nn.vae import VAE, binary_cross_entropy_with_logits, kl_divergence, vae_loss
from flax import nnx


@pytest.fixture
def vae() -> VAE:
	"""Create a VAE model for testing."""
	spatial_dims = (28, 28)
	features = (1, 32, 64)
	latent_size = 10
	rngs = nnx.Rngs(0)
	return VAE(spatial_dims, features, latent_size, rngs)


def test_vae_initialization(vae: VAE) -> None:
	"""Test the initialization of the VAE model."""
	assert isinstance(vae, VAE)
	assert isinstance(vae.encoder, nnx.Module)
	assert isinstance(vae.decoder, nnx.Module)


def test_vae_encode(vae: VAE) -> None:
	"""Test the encode method of the VAE model."""
	key = jax.random.key(0)
	x = jax.random.normal(key, (1, 28, 28, 1))
	z, mean, logvar = vae.encode(x)
	assert z.shape == (1, 10)
	assert mean.shape == (1, 10)
	assert logvar.shape == (1, 10)


def test_vae_decode(vae: VAE) -> None:
	"""Test the decode method of the VAE model."""
	key = jax.random.key(0)
	z = jax.random.normal(key, (1, 10))
	logits = vae.decode(z)
	assert logits.shape == (1, 28, 28, 1)


def test_vae_generate(vae: VAE) -> None:
	"""Test the generate method of the VAE model."""
	key = jax.random.key(0)
	z = jax.random.normal(key, (1, 10))
	generated = vae.generate(z)
	assert generated.shape == (1, 28, 28, 1)
	assert jnp.all((generated >= 0) & (generated <= 1))


def test_vae_forward(vae: VAE) -> None:
	"""Test the forward pass of the VAE model."""
	key = jax.random.key(0)
	x = jax.random.normal(key, (1, 28, 28, 1))
	logits, mean, logvar = vae(x)
	assert logits.shape == (1, 28, 28, 1)
	assert mean.shape == (1, 10)
	assert logvar.shape == (1, 10)


def test_kl_divergence() -> None:
	"""Test the KL divergence calculation."""
	mean = jnp.array([0.0, 1.0, -1.0])
	logvar = jnp.array([0.0, 0.5, -0.5])
	kl_div = kl_divergence(mean, logvar)
	assert kl_div.shape == ()
	assert kl_div >= 0


def test_binary_cross_entropy_with_logits() -> None:
	"""Test the binary cross-entropy with logits calculation."""
	logits = jnp.array([-1.0, 0.0, 1.0])
	labels = jnp.array([0.0, 0.5, 1.0])
	bce = binary_cross_entropy_with_logits(logits, labels)
	assert bce.shape == ()
	assert bce >= 0


def test_vae_loss() -> None:
	"""Test the VAE loss calculation."""
	logits = jnp.array([[[-1.0, 0.0, 1.0]]])
	targets = jnp.array([[[0.0, 0.5, 1.0]]])
	mean = jnp.array([[0.0, 1.0, -1.0]])
	logvar = jnp.array([[0.0, 0.5, -0.5]])
	loss = vae_loss(logits, targets, mean, logvar)
	assert loss.shape == ()
	assert loss >= 0
