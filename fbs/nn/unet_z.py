import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn.base import sinusoidal_embedding
from typing import Callable, Sequence


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    use_bias: bool = False

    @nn.compact
    def __call__(self, img):
        batch, h, w, channels = img.shape
        img = img.reshape(batch, h * w, channels)
        batch, n, channels = img.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(self.dim * 3, use_bias=self.use_bias, kernel_init=nn.initializers.xavier_uniform())(img)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attention = nn.softmax(attention, axis=-1)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = jnp.reshape(x, (batch, int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5), -1))
        return x


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
        z = nn.Conv(self.nfeatures, kernel_size=(1, 1))(x)
        return y + z


class MNISTUNet(nn.Module):
    dt: float
    features: Sequence[int] = (16, 32, 64)
    nchannels = 1

    @nn.compact
    def __call__(self, x, t):
        # x: (n, 784) or (n, 28, 28) or (784, )
        # t: float
        if x.ndim <= 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        x = x.reshape(batch_size, 28, 28, self.nchannels)

        # Top
        x = nn.Conv(features=16, kernel_size=(7, 7), padding=((3, 3), (3, 3)))(x)
        time_emb = TimeEmbedding(self.dt)(t)

        # Down pass
        up_layers = []
        for i, feature in enumerate(self.features):
            x = ResBlock(feature)(x, time_emb)
            a = Attention(feature)(x)
            n = nn.GroupNorm(num_groups=4)(a)
            x = x + n
            up_layers.append(x)
            if i < len(self.features) - 1:
                x = nn.Conv(feature, kernel_size=(3, 3), strides=(2, 2))(x)

        # Middle
        x = ResBlock(self.features[-1])(x, time_emb)
        a = Attention(self.features[-1])(x)
        n = nn.GroupNorm(num_groups=4)(a)
        x = x + n
        x = ResBlock(self.features[-1])(x, time_emb)

        # Up pass
        for i, feature in reversed(list(enumerate(self.features))):
            x = jnp.concatenate([up_layers[i], x], -1)
            x = ResBlock(feature)(x, time_emb)
            a = Attention(feature)(x)
            n = nn.GroupNorm(num_groups=4)(a)
            x = x + n
            if i > 0:
                # x = nn.ConvTranspose(feature, kernel_size=(3, 3), strides=(2, 2))(x)
                x = jax.image.resize(x, (batch_size, x.shape[1] * 2, x.shape[2] * 2, feature), 'nearest')

        # End
        x = ResBlock(16)(x, time_emb)
        x = nn.Conv(self.nchannels, kernel_size=(1, 1))(x)
        return jnp.squeeze(jnp.reshape(x, (batch_size, -1)))
