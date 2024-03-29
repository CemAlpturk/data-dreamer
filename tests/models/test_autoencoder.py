import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn
import pytest

from data_dreamer.models import AutoEncoder


def test_autoencoder_forward_pass() -> None:
    input_dim = 10
    latent_dim = 5
    hidden_dims = [20, 15]
    activation = jnn.relu
    final_activation = jnn.sigmoid

    model = AutoEncoder(
        input_dim, latent_dim, hidden_dims, activation, final_activation
    )

    x = jnp.ones((input_dim,))
    reconstructed_x = model(x)

    assert reconstructed_x.shape == (input_dim,)


def test_autoencoder_initialization() -> None:
    input_dim = 10
    latent_dim = 5
    hidden_dims = [20, 15]
    activation = jnn.relu
    final_activation = jnn.sigmoid

    model = AutoEncoder(
        input_dim, latent_dim, hidden_dims, activation, final_activation
    )

    assert len(model.encoder.layers) == len(hidden_dims) * 2 + 2
    assert len(model.decoder.layers) == len(hidden_dims) * 2 + 2


def test_autoencoder_random_key() -> None:
    input_dim = 10
    latent_dim = 5
    hidden_dims = [20, 15]
    activation = jnn.relu
    final_activation = jnn.sigmoid

    model1 = AutoEncoder(
        input_dim, latent_dim, hidden_dims, activation, final_activation
    )
    model2 = AutoEncoder(
        input_dim, latent_dim, hidden_dims, activation, final_activation
    )

    assert model1.encoder != model2.encoder
    assert model1.decoder != model2.decoder


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", str(__file__)])
