import math
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn.base import sinusoidal_embedding
from typing import Callable, Sequence


class ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(num_groups=4)(x)
        x = nn.silu(x)
        return x


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, time_emb):
        x0 = ConvBlock(self.features)(x)

        x = ConvBlock(self.features)(x)
        time_emb = nn.silu(time_emb)
        time_emb = nn.Dense(self.features)(time_emb)
        x = x + jnp.reshape(time_emb, (1, 1, 1, -1))

        x0 = nn.Conv(self.features, kernel_size=(1, 1))(x0)
        return x + x0


class MNISTUnet(nn.Module):
    dt: float
    features: Sequence[int] = (16, 32, 64, )
    nchannels = 1

    @nn.compact
    def __call__(self, x, t):
        # x: (n, 784) or (n, 28, 28)
        # t: float
        if x.ndim <= 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        x = x.reshape(batch_size, 28, 28, self.nchannels)

        # Top
        x = nn.Conv(features=16, kernel_size=(7, 7), padding=((3, 3), (3, 3)))(x)
        time_emb = sinusoidal_embedding(t / self.dt, out_dim=16)

        # Down pass
        layers = []
        for feature in self.features:
            x = ResBlock(feature)(x, time_emb)
            a = Attention(feature, x)
            n = nn.GroupNorm(num_groups=4)(a)
            x = x + n
            x = nn.Conv(feature, kernel_size=(4, 4), strides=(2, 2))(x)
            layers.append(x)

        # Middle
        x = ResBlock(self.features[-1])(x, time_emb)
        a = Attention(x)
        n = nn.GroupNorm(num_groups=4)(a)
        x = x + n
        x = ResBlock(self.features[-1])(x, time_emb)

        # Up pass
        for i, feature in reversed(list(enumerate(self.features))):
            x = jnp.concatenate([layers[i], x], -1)
            x = ResBlock(feature)(x, time_emb)
            a = Attention(x)
            n = nn.GroupNorm(num_groups=4)(a)
            x = x + n
            x = nn.ConvTranspose(feature, kernel_size=(4, 4), strides=(2, 2))(x)

        # End
        x = ResBlock(16)(x, time_emb)
        x = nn.Conv(self.nchannels, kernel_size=(1, 1))(x)
        return jnp.squeeze(jnp.reshape(x, (batch_size, -1)))
