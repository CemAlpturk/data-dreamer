from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


class MLP(eqx.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.
        hidden_dims (Iterable[int], optional): The dimensions of the hidden layers. Defaults to ().
        activation (Callable | None, optional): The activation function to use for the hidden layers. Defaults to None.
        final_activation (Callable | None, optional): The activation function to use for the final output layer. Defaults to None.
        key (jax.Array | None, optional): The random key for initializing the model parameters. Defaults to None.

    Attributes:
        layers (list[Callable]): The list of layers in the MLP.

    Methods:
        __call__(self, x: jnp.ndarray) -> jnp.ndarray: Forward pass of the MLP.

    """

    layers: list[Callable]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (),
        activation: Callable | None = None,
        final_activation: Callable | None = None,
        *,
        key: jax.Array | None = None,
    ) -> None:
        if key is None:
            key = jrandom.PRNGKey(42)

        dims = [input_dim, *hidden_dims, output_dim]
        keys = jrandom.split(key, len(dims) - 1)

        if activation is None:
            activation = lambda x: x
        if final_activation is None:
            final_activation = lambda x: x

        layers = []

        for i in range(len(dims) - 1):
            layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))
            if i < len(dims) - 2:
                layers.append(activation)
            else:
                layers.append(final_activation)

        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
