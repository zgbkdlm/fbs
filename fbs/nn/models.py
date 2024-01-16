import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn import sinusoidal_embedding, make_st_nn

nn_param_init = nn.initializers.xavier_normal()
nn_param_dtype = jnp.float64


class MNISTAutoEncoder(nn.Module):
    @nn.compact
    def __call__(self, xy, t):
        xy = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(xy)
        xy = nn.relu(xy)
        xy = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(xy)

        t = sinusoidal_embedding(t, out_dim=128)
        t = nn.Dense(features=64, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
        t = nn.relu(t)
        t = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)

        z = jnp.concatenate([xy, t], axis=-1)
        z = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=256, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=784 * 2, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        return jnp.squeeze(z)


class MNISTConv(nn.Module):
    @nn.compact
    def __call__(self, xy, t):
        xy = jnp.concatenate([xy[..., :784].reshape(-1, 28, 28, 1),
                              xy[..., 784:].reshape(-1, 28, 28, 1)], axis=-1)  # (n, 28, 28, 2)
        batch_size = xy.shape[0]
        xy = nn.Conv(features=32, kernel_size=(3, 3))(xy)
        xy = nn.relu(xy)
        xy = nn.avg_pool(xy, window_shape=(2, 2), strides=(2, 2))
        xy = nn.Conv(features=64, kernel_size=(3, 3))(xy)
        xy = nn.relu(xy)
        xy = nn.avg_pool(xy, window_shape=(2, 2), strides=(2, 2))
        xy = xy.reshape((xy.shape[0], -1))
        xy = nn.Dense(features=256)(xy)
        xy = nn.relu(xy)
        xy = nn.Dense(features=64)(xy)

        t = sinusoidal_embedding(t, out_dim=128)
        t = nn.Dense(features=64, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
        t = nn.relu(t)
        t = nn.Dense(features=64, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
        t = t.reshape(batch_size, -1)

        z = jnp.concatenate([xy, t], axis=-1)
        z = nn.Dense(features=256, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=784 * 2, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        return jnp.squeeze(z)


def make_simple_st_nn(key, dim_in, batch_size, mlp: nn.Module = None,
                      embed_dim: int = 128):
    """Make a simple spatio-temporal neural network with sinusoidal embedding.

    Returns
    -------

    """

    class ClassicMLP(nn.Module):
        @nn.compact
        def __call__(self, x, t):
            d = x.shape[-1]

            # Spatial part
            x = nn.Dense(features=16, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
            x = nn.relu(x)
            x = nn.Dense(features=8, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)

            # Temporal part
            t = sinusoidal_embedding(t, out_dim=embed_dim)
            t = nn.Dense(features=16, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
            t = nn.relu(t)
            t = nn.Dense(features=8, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)

            z = jnp.concatenate([x, t], axis=-1)
            z = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
            z = nn.relu(z)
            z = nn.Dense(features=8, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
            z = nn.relu(z)
            z = nn.Dense(features=d, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
            return jnp.squeeze(z)

    if mlp is None:
        mlp = ClassicMLP()
    dict_param = mlp.init(key, jnp.ones((batch_size, dim_in)), jnp.ones((batch_size, 1)))
    array_param, array_to_dict, forward_pass = make_st_nn(mlp, dim_in, batch_size, key)
    return mlp, dict_param, array_param, array_to_dict, forward_pass


# TODO: MLP-mixer https://github.com/google-research/big_vision/blob/main/big_vision/models/mlp_mixer.py
