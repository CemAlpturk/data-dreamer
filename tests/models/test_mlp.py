import jax.nn as jnn
import jax.numpy as jnp
import pytest

from data_dreamer.models import MLP


def test_mlp_forward_pass() -> None:
    input_dim = 10
    output_dim = 5
    hidden_dims = [20, 15]
    activation = jnn.relu
    final_activation = jnn.sigmoid

    model = MLP(input_dim, output_dim, hidden_dims, activation, final_activation)

    x = jnp.ones((input_dim,))
    y = model(x)

    assert y.shape == (output_dim,)


def test_mlp_initialization() -> None:
    input_dim = 10
    output_dim = 5
    hidden_dims = [20, 15]
    activation = jnn.relu
    final_activation = jnn.sigmoid

    model = MLP(input_dim, output_dim, hidden_dims, activation, final_activation)

    assert len(model.layers) == len(hidden_dims) * 2 + 2


def test_mlp_random_key() -> None:
    input_dim = 10
    output_dim = 5
    hidden_dims = [20, 15]
    activation = jnn.relu
    final_activation = jnn.sigmoid

    model1 = MLP(input_dim, output_dim, hidden_dims, activation, final_activation)
    model2 = MLP(input_dim, output_dim, hidden_dims, activation, final_activation)

    assert model1 == model2


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", str(__file__)])
