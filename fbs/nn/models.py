import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn import sinusoidal_embedding, make_st_nn
from fbs.nn.utils import PixelShuffle
from typing import Sequence

nn_param_dtype = jnp.float64
nn_param_init = nn.initializers.xavier_normal()


class _CrescentTimeBlock(nn.Module):
    dt: float
    nfeatures: int

    @nn.compact
    def __call__(self, time_emb):
        time_emb = nn.Dense(features=self.nfeatures, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(time_emb)
        time_emb = nn.gelu(time_emb)
        time_emb = nn.Dense(features=self.nfeatures, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(time_emb)
        return time_emb


class CrescentMLP(nn.Module):
    dt: float

    @nn.compact
    def __call__(self, x, t):
        if t.ndim < 1:
            time_emb = jnp.expand_dims(sinusoidal_embedding(t / self.dt, out_dim=32), 0)
        else:
            time_emb = jax.vmap(lambda z: sinusoidal_embedding(z, out_dim=32))(t / self.dt)

        x = nn.Dense(features=64, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
        x = (x * _CrescentTimeBlock(dt=self.dt, nfeatures=64)(time_emb) +
             _CrescentTimeBlock(dt=self.dt, nfeatures=64)(time_emb))
        x = nn.gelu(x)
        x = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
        x = (x * _CrescentTimeBlock(dt=self.dt, nfeatures=32)(time_emb) +
             _CrescentTimeBlock(dt=self.dt, nfeatures=32)(time_emb))
        x = nn.gelu(x)
        x = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=3, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)

        return jnp.squeeze(x)


class _GMSBMLPResBlock(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, time_emb):
        time_emb = nn.Dense(features=2 * self.dim, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(time_emb)
        time_emb = nn.swish(time_emb)
        scale, shift = jnp.split(time_emb, 2, axis=-1)

        x = nn.Dense(features=self.dim, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = x * (1 + scale) + shift
        x = nn.Dense(features=self.dim, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        return x


class GMSBMLP(nn.Module):
    """Ad-hoc nn for the Schrodinger bridge."""
    dim: int

    @nn.compact
    def __call__(self, x, k):
        if k.ndim < 1:
            time_emb = jnp.expand_dims(sinusoidal_embedding(k, out_dim=32), 0)
        else:
            time_emb = jax.vmap(lambda z: sinusoidal_embedding(z, out_dim=32))(k)

        x0 = x

        # x1 = _GMSBMLPResBlock(dim=16)(x, time_emb)
        # x2 = _GMSBMLPResBlock(dim=32)(x1, time_emb)
        # x3 = _GMSBMLPResBlock(dim=64)(x2, time_emb)
        #
        # x3_ = x3 + _GMSBMLPResBlock(dim=64)(x3, time_emb)
        # x2_ = x2 + _GMSBMLPResBlock(dim=32)(x3_, time_emb)
        # x1_ = x1 + _GMSBMLPResBlock(dim=16)(x2_, time_emb)

        x1 = _GMSBMLPResBlock(dim=16)(x, time_emb)
        x2 = _GMSBMLPResBlock(dim=64)(x1, time_emb)

        x2_ = x2 + _GMSBMLPResBlock(dim=64)(x2, time_emb)
        x1_ = x1 + _GMSBMLPResBlock(dim=16)(x2_, time_emb)

        x = x0 + nn.Dense(features=self.dim, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x1_)
        return x


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
        # here add attention
        x1 = x
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (n, 14, 14, 32)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)  # (n, 14, 14, 64)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.silu(x)
        x2 = x
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (n, 7, 7, 64)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)

        t = sinusoidal_embedding(t / self.dt, out_dim=32)
        t = nn.Dense(features=64, param_dtype=self.nn_param_dtype, kernel_init=nn.initializers.xavier_normal())(t)
        t = nn.gelu(t)
        t = nn.Dense(features=128, param_dtype=self.nn_param_dtype, kernel_init=nn.initializers.xavier_normal())(t)
        t = t.reshape(1, 1, 1, -1)

        t1, t2 = t[:, :, :, :64], t[:, :, :, 64:]

        x = x * t1 + t2  # (n, 7, 7, 64)

        if self.decoder == 'pixel_shuffle':
            x = nn.Conv(features=64 * 4, kernel_size=(3, 3))(x)  # (n, 7, 7, 64 * 4)
            x = PixelShuffle(scale=2)(x)  # (n, 14, 14, 64)
            x = nn.Conv(features=64, kernel_size=(3, 3))(x)  # (n, 14, 14, 64)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            # Add attention
            x = x + x2
            x = nn.Conv(features=32 * 4, kernel_size=(3, 3))(x)  # (n, 14, 14, 32 * 4)
            x = PixelShuffle(scale=2)(x)  # (n, 28, 28, 32)
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)  # (n, 28, 28, 32)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x1
            x = nn.Conv(features=1, kernel_size=(3, 3))(x)  # (n, 28, 28, 1)
        else:
            x = jax.image.resize(x, (self.batch_spatial, 14, 14, 64), 'nearest')
            x = nn.Conv(features=64, kernel_size=(3, 3))(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x2
            x = jax.image.resize(x, (self.batch_spatial, 28, 28, 64), 'nearest')
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.silu(x)
            x = x + x1
            x = nn.Conv(features=1, kernel_size=(1, 1))(x)

        x = x.reshape((batch_size, -1))
        return jnp.squeeze(x)


def make_simple_st_nn(key,
                      dim_in: Sequence[int],
                      batch_size, nn_model: nn.Module = None,
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
    dict_param = nn_model.init(key, jnp.ones((batch_size, *dim_in)), jnp.array(1.))
    array_param, array_to_dict, forward_pass = make_st_nn(key, nn_model, dim_in, batch_size)
    return nn_model, dict_param, array_param, array_to_dict, forward_pass

# TODO: MLP-mixer https://github.com/google-research/big_vision/blob/main/big_vision/models/mlp_mixer.py
