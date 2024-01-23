import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from fbs.nn.utils import PixelShuffle
from fbs.nn.base import sinusoidal_embedding
from typing import List, Sequence


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
# class Attention(nn.Module):
#     """Self-attention residual block.
#     https://github.com/google-research/vdm/blob/main/model_vdm.py#L468
#     """
#     num_heads: int
#
#     @nn.compact
#     def __call__(self, x):
#         B, H, W, C = x.shape
#
#         h = nn.GroupNorm(num_groups=4)(x)
#         if self.num_heads == 1:
#             q = nn.Dense(features=C, name='q')(h)
#             k = nn.Dense(features=C, name='k')(h)
#             v = nn.Dense(features=C, name='v')(h)
#             h = dot_product_attention(
#                 q[:, :, :, None, :],
#                 k[:, :, :, None, :],
#                 v[:, :, :, None, :],
#                 axis=(1, 2))[:, :, :, 0, :]
#             h = nn.Dense(
#                 features=C, kernel_init=nn.initializers.zeros, name='proj_out')(h)
#         else:
#             head_dim = C // self.num_heads
#             q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
#             k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
#             v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
#             assert q.shape == k.shape == v.shape == (
#                 B, H, W, self.num_heads, head_dim)
#             h = dot_product_attention(q, k, v, axis=(1, 2))
#             h = nn.DenseGeneral(
#                 features=C,
#                 axis=(-2, -1),
#                 kernel_init=nn.initializers.zeros,
#                 name='proj_out')(h)
#
#         return x + h


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
    features: Sequence[int] = (16, 32, 64)
    nchannels = 1
    upsampling_method: str = 'pixel_shuffle'

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
            # a = Attention(feature)(x)
            # n = nn.GroupNorm(num_groups=4)(a)
            # x = x + n
            x = x + nn.GroupNorm(num_groups=4)(x)
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
        down_features = (16,) + self.features[:-1]
        for i in reversed(range(len(self.features))):
            x = jnp.concatenate([up_layers[i], x], -1)
            x = ResBlock(down_features[i])(x, time_emb)
            # a = Attention(down_features[i])(x)
            # n = nn.GroupNorm(num_groups=4)(a)
            # x = x + n
            x = x + nn.GroupNorm(num_groups=4)(x)
            if i > 0:
                if self.upsampling_method == 'conv':
                    x = nn.ConvTranspose(down_features[i], kernel_size=(3, 3), strides=(2, 2))(x)
                elif self.upsampling_method == 'resize':
                    x = jax.image.resize(x, (batch_size, x.shape[1] * 2, x.shape[2] * 2, down_features[i]), 'nearest')
                elif self.upsampling_method == 'pixel_shuffle':
                    x = nn.Conv(features=down_features[i] * 4, kernel_size=(3, 3))(x)
                    x = PixelShuffle(scale=2)(x)
                    x = nn.Conv(features=down_features[i], kernel_size=(3, 3))(x)

        # End
        x = ResBlock(16)(x, time_emb)
        x = nn.Conv(self.nchannels, kernel_size=(1, 1))(x)
        return jnp.squeeze(jnp.reshape(x, (batch_size, -1)))
