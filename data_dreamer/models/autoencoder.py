from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx

from data_dreamer.models import MLP


class AutoEncoder(eqx.Module):
    encoder: MLP
    decoder: MLP

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Iterable[int] = (),
        activation: Callable | None = None,
        final_activation: Callable | None = None,
        key: jax.Array | None = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(42)

        encoder_key, decoder_key = jrandom.split(key)

        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            final_activation=final_activation,
            key=encoder_key,
        )

        self.decoder = MLP(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims[::-1],
            activation=activation,
            final_activation=final_activation,
            key=decoder_key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(self.encoder(x))
