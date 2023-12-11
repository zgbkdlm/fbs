import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn import sinusoidal_embedding, make_st_nn

nn_param_init = nn.initializers.xavier_normal()


def make_simple_st_nn(key, dim_x, batch_size,
                      embed_dim: int = 128):
    """Make a simple spatio-temporal neural network with sinusoidal embedding.

    Returns
    -------

    """

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x, t):
            d = x.shape[-1]

            # Spatial part
            x = nn.Dense(features=16, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
            x = nn.relu(x)
            x = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)

            # Temporal part
            t = sinusoidal_embedding(t, out_dim=embed_dim)
            t = nn.Dense(features=16, param_dtype=jnp.float64, kernel_init=nn_param_init)(t)
            t = nn.relu(t)
            t = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(t)

            z = jnp.concatenate([x, t], axis=-1)
            z = nn.Dense(features=32, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
            z = nn.relu(z)
            z = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
            z = nn.relu(z)
            z = nn.Dense(features=d, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)

            return jnp.squeeze(z)

    mlp = MLP()
    dict_param = mlp.init(key, jnp.ones((batch_size, dim_x)), jnp.ones((batch_size, 1)))
    array_param, array_to_dict, forward_pass = make_st_nn(mlp, dim_x, batch_size, key)
    return mlp, dict_param, array_param, array_to_dict, forward_pass
