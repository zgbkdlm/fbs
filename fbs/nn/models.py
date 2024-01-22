import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn import sinusoidal_embedding, make_st_nn
from fbs.nn.utils import PixelShuffle


class MNISTAutoEncoder(nn.Module):
    # This does not really work.
    nn_param_dtype = jnp.float64
    nn_param_init = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, xy, t):
        xy = nn.Dense(features=128, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(xy)
        xy = nn.relu(xy)
        xy = nn.Dense(features=32, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(xy)

        t = sinusoidal_embedding(t, out_dim=128)
        t = nn.Dense(features=64, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(t)
        t = nn.relu(t)
        t = nn.Dense(features=32, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(t)

        z = jnp.concatenate([xy, t], axis=-1)
        z = nn.Dense(features=128, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=256, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=784 * 2, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(z)
        return jnp.squeeze(z)


class MNISTResConv(nn.Module):
    nn_param_dtype = jnp.float64
    nn_param_init = nn.initializers.xavier_normal()
    dt: float
    decoder: str = 'pixel_shuffle'

    @nn.compact
    def __call__(self, x, t):
        # x: (n, 784) or (n, 28, 28)
        # t: float
        if x.ndim <= 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        x = x.reshape(batch_size, 28, 28, 1)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)  # (n, 28, 28, 32)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.silu(x)
        x1 = x
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (n, 14, 14, 32)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)  # (n, 14, 14, 64)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.silu(x)
        x2 = x
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (n, 7, 7, 64)

        t = sinusoidal_embedding(t / self.dt, out_dim=32)
        t = nn.Dense(features=64, param_dtype=self.nn_param_dtype, kernel_init=nn.initializers.xavier_normal())(t)
        t = nn.gelu(t)
        t = nn.Dense(features=128, param_dtype=self.nn_param_dtype, kernel_init=nn.initializers.xavier_normal())(t)
        t = t.reshape(1, 1, 1, -1)

        t1, t2 = t[:, :, :, :64], t[:, :, :, 64:]

        x = x * t1 + t2  # (n, 7, 7, 64)

        if self.decoder == 'pixel_shuffle':
            x = PixelShuffle(scale=2)(x)  # (n, 14, 14, 16)
            x = nn.Conv(features=64, kernel_size=(2, 2))(x)  # (n, 14, 14, 64)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x2
            x = PixelShuffle(scale=2)(x)  # (n, 28, 28, 16)
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)  # (n, 28, 28, 32)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x1
            x = nn.Conv(features=1, kernel_size=(2, 2))(x)  # (n, 28, 28, 1)
        else:
            x = jax.image.resize(x, (self.batch_spatial, 14, 14, 64), 'nearest')
            x = nn.Conv(features=64, kernel_size=(2, 2))(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x2
            x = jax.image.resize(x, (self.batch_spatial, 28, 28, 64), 'nearest')
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x1
            x = nn.Conv(features=1, kernel_size=(2, 2))(x)

        x = x.reshape((batch_size, -1))
        return jnp.squeeze(x)


def make_simple_st_nn(key, dim_in, batch_size, nn_model: nn.Module = None,
                      embed_dim: int = 128):
    """Make a simple spatio-temporal neural network with sinusoidal embedding.

    Returns
    -------

    """

    class ClassicMLP(nn.Module):
        nn_param_dtype = jnp.float64
        nn_param_init = nn.initializers.xavier_normal()

        @nn.compact
        def __call__(self, x, t):
            d = x.shape[-1]

            # Spatial part
            x = nn.Dense(features=16, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(x)
            x = nn.relu(x)
            x = nn.Dense(features=8, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(x)

            # Temporal part
            t = sinusoidal_embedding(t, out_dim=embed_dim)
            t = nn.Dense(features=16, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(t)
            t = nn.relu(t)
            t = nn.Dense(features=8, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(t)

            z = jnp.concatenate([x, t], axis=-1)
            z = nn.Dense(features=32, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(z)
            z = nn.relu(z)
            z = nn.Dense(features=8, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(z)
            z = nn.relu(z)
            z = nn.Dense(features=d, param_dtype=self.nn_param_dtype, kernel_init=self.nn_param_init)(z)
            return jnp.squeeze(z)

    if nn_model is None:
        nn_model = ClassicMLP()
    dict_param = nn_model.init(key, jnp.ones((batch_size, dim_in)), jnp.array(1.))
    array_param, array_to_dict, forward_pass = make_st_nn(nn_model, dim_in, batch_size, key)
    return nn_model, dict_param, array_param, array_to_dict, forward_pass

# TODO: MLP-mixer https://github.com/google-research/big_vision/blob/main/big_vision/models/mlp_mixer.py
