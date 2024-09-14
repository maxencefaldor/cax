"""Tests for the VAE module."""

import jax
import jax.numpy as jnp
import pytest
from cax.nn.vae import VAE, binary_cross_entropy_with_logits, kl_divergence, vae_loss
from flax import nnx


@pytest.fixture
def vae_model():
	"""Create a VAE model for testing."""
	spatial_dims = (28, 28)
	features = (1, 32, 64)
	latent_size = 10
	rngs = nnx.Rngs(0)
	return VAE(spatial_dims, features, latent_size, rngs)


def test_vae_initialization(vae_model):
	"""Test the initialization of the VAE model."""
	assert isinstance(vae_model, VAE)
	assert isinstance(vae_model.encoder, nnx.Module)
	assert isinstance(vae_model.decoder, nnx.Module)


def test_vae_encode(vae_model):
	"""Test the encode method of the VAE model."""
	key = jax.random.PRNGKey(0)
	x = jax.random.normal(key, (1, 28, 28, 1))
	z, mean, logvar = vae_model.encode(x, key)
	assert z.shape == (1, 10)
	assert mean.shape == (1, 10)
	assert logvar.shape == (1, 10)


def test_vae_decode(vae_model):
	"""Test the decode method of the VAE model."""
	key = jax.random.PRNGKey(0)
	z = jax.random.normal(key, (1, 10))
	logits = vae_model.decode(z)
	assert logits.shape == (1, 28, 28, 1)


def test_vae_generate(vae_model):
	"""Test the generate method of the VAE model."""
	key = jax.random.PRNGKey(0)
	z = jax.random.normal(key, (1, 10))
	generated = vae_model.generate(z)
	assert generated.shape == (1, 28, 28, 1)
	assert jnp.all((generated >= 0) & (generated <= 1))


def test_vae_forward(vae_model):
	"""Test the forward pass of the VAE model."""
	key = jax.random.PRNGKey(0)
	x = jax.random.normal(key, (1, 28, 28, 1))
	logits, mean, logvar = vae_model(x, key)
	assert logits.shape == (1, 28, 28, 1)
	assert mean.shape == (1, 10)
	assert logvar.shape == (1, 10)


def test_kl_divergence():
	"""Test the KL divergence calculation."""
	mean = jnp.array([0.0, 1.0, -1.0])
	logvar = jnp.array([0.0, 0.5, -0.5])
	kl_div = kl_divergence(mean, logvar)
	assert kl_div.shape == ()
	assert kl_div >= 0


def test_binary_cross_entropy_with_logits():
	"""Test the binary cross-entropy with logits calculation."""
	logits = jnp.array([-1.0, 0.0, 1.0])
	labels = jnp.array([0.0, 0.5, 1.0])
	bce = binary_cross_entropy_with_logits(logits, labels)
	assert bce.shape == ()
	assert bce >= 0


def test_vae_loss():
	"""Test the VAE loss calculation."""
	logits = jnp.array([[[-1.0, 0.0, 1.0]]])
	targets = jnp.array([[[0.0, 0.5, 1.0]]])
	mean = jnp.array([[0.0, 1.0, -1.0]])
	logvar = jnp.array([[0.0, 0.5, -0.5]])
	loss = vae_loss(logits, targets, mean, logvar)
	assert loss.shape == ()
	assert loss >= 0


def test_vae_vmap(vae_model):
	"""Test the vectorized mapping of the VAE model."""
	key = jax.random.PRNGKey(0)
	batch_size = 4
	x = jax.random.normal(key, (batch_size, 28, 28, 1))
	keys = jax.random.split(key, batch_size)

	logits, mean, logvar = nnx.vmap(vae_model)(x, keys)

	assert logits.shape == (batch_size, 28, 28, 1)
	assert mean.shape == (batch_size, 10)
	assert logvar.shape == (batch_size, 10)
