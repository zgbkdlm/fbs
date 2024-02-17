"""
The Attention block is adapted from
https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/models/layers.py#L496.
"""
import string
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn.utils import PixelShuffle
from fbs.nn.base import sinusoidal_embedding
from typing import List, Sequence


def _einsum(a, b, c, x, y):
    """Due to Song Yang"""
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1). Due to Song Yang"""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_uppercase[:len(y.shape)])
    assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


def default_init(scale=1.):
    """The same initialization used in DDPM. Due to Song Yang"""
    scale = 1e-10 if scale == 0 else scale
    return jax.nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class NIN(nn.Module):
    """Due to Song Yang"""
    num_units: int
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, x):
        in_dim = int(x.shape[-1])
        W = self.param('W', default_init(scale=self.init_scale), (in_dim, self.num_units))
        b = self.param('b', jax.nn.initializers.zeros, (self.num_units,))
        y = contract_inner(x, W) + b
        assert y.shape == x.shape[:-1] + (self.num_units,)
        return y


class Attention(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM. Due to Song Yang"""
    skip_rescale: bool = False
    init_scale: float = 0.

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
        q = NIN(C)(h)
        k = NIN(C)(h)
        v = NIN(C)(h)

        w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
        w = jnp.reshape(w, (B, H, W, H * W))
        w = jax.nn.softmax(w, axis=-1)
        w = jnp.reshape(w, (B, H, W, H, W))
        h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
        h = NIN(C, init_scale=self.init_scale)(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / math.sqrt(2.)


class TimeEmbedding(nn.Module):
    dt: float
    dim: int = 16

    @nn.compact
    def __call__(self, t):
        t = sinusoidal_embedding(t / self.dt, out_dim=self.dim)
        t = nn.Dense(self.dim * 2)(t)
        t = nn.gelu(t)
        t = nn.Dense(self.dim * 2)(t)
        return t


class ConvBlock(nn.Module):
    nfeatures: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.nfeatures, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(num_groups=4)(x)
        x = nn.silu(x)
        return x


class ResBlock(nn.Module):
    nfeatures: int

    @nn.compact
    def __call__(self, x, time_emb):
        y = ConvBlock(self.nfeatures)(x)
        time_emb = nn.silu(time_emb)
        time_emb = nn.Dense(self.nfeatures)(time_emb)
        y = y + jnp.reshape(time_emb, (1, 1, 1, -1))  # TODO
        y = ConvBlock(self.nfeatures)(y)
        z = nn.Conv(self.nfeatures, kernel_size=(3, 3))(x)
        return y + z


class MNISTUNet(nn.Module):
    dt: float
    features: Sequence[int] = (32, 64, 128)
    nchannels: int = 1
    upsampling_method: str = 'pixel_shuffle'

    @nn.compact
    def __call__(self, x, t):
        # x: (n, ...) or (..., )
        # t: float
        if x.ndim == 4:
            batch_size = x.shape[0]
        else:
            batch_size = 1
            x = jnp.expand_dims(x, 0)

        # Top
        x = nn.Conv(features=16, kernel_size=(7, 7), padding=((3, 3), (3, 3)))(x)
        time_emb = TimeEmbedding(self.dt)(t)

        # Down pass
        up_layers = []
        for i, feature in enumerate(self.features):
            x = ResBlock(feature)(x, time_emb)
            # x = Attention()(x)
            up_layers.append(x)
            if i < len(self.features) - 1:
                x = nn.Conv(feature, kernel_size=(3, 3), strides=(2, 2))(x)

        # Middle
        x = ResBlock(128)(x, time_emb)
        x = Attention()(x)

        # Up pass
        for i in reversed(range(len(self.features))):
            x = ResBlock(self.features[i])(x, time_emb)
            # x = Attention()(x)
            x = jnp.concatenate([up_layers[i], x], -1)
            x = ResBlock(self.features[i])(x, time_emb)
            if i > 0:
                if self.upsampling_method == 'conv':
                    x = nn.ConvTranspose(self.features[i - 1], kernel_size=(3, 3), strides=(2, 2))(x)
                elif self.upsampling_method == 'resize':
                    x = jax.image.resize(x, (batch_size, x.shape[1] * 2, x.shape[2] * 2, self.features[i - 1]),
                                         'nearest')
                    x = nn.Conv(features=self.features[i - 1], kernel_size=(3, 3))(x)
                elif self.upsampling_method == 'pixel_shuffle':
                    x = nn.Conv(features=self.features[i - 1] * 4, kernel_size=(3, 3))(x)
                    x = PixelShuffle(scale=2)(x)
                    x = nn.Conv(features=self.features[i - 1], kernel_size=(3, 3))(x)
                else:
                    raise NotImplementedError('...')

        # End
        x = ResBlock(8)(x, time_emb)
        x = nn.Conv(self.nchannels, kernel_size=(1, 1))(x)

        if batch_size == 1:
            return x[0]
        else:
            return x
