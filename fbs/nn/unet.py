"""
Notice
======
The implementation in this script is modified from
https://www.kaggle.com/code/darshan1504/exploring-diffusion-models-with-jax under the Apache 2.0 license.
"""
import math
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, img):
        batch, h, w, channels = img.shape
        img = img.reshape(batch, h * w, channels)
        batch, n, channels = img.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(self.dim * 3, use_bias=self.use_bias, kernel_init=self.kernel_init)(img)
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


class SinusoidalEmbedding(nn.Module):
    dim: int = 32

    @nn.compact
    def __call__(self, inputs):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb


class TimeEmbedding(nn.Module):
    dim: int = 32

    @nn.compact
    def __call__(self, t):
        time_dim = self.dim * 4

        se = SinusoidalEmbedding(self.dim)(t)

        # Projecting the embedding into a 128 dimensional space
        x = nn.Dense(time_dim)(se)
        x = nn.gelu(x)
        x = nn.Dense(time_dim)(x)

        return x


class Block(nn.Module):
    dim: int = 32
    groups: int = 8

    @nn.compact
    def __call__(self, inputs):
        conv = nn.Conv(self.dim, (3, 3))(inputs)
        norm = nn.GroupNorm(num_groups=self.groups)(conv)
        activation = nn.silu(norm)
        return activation


class ResnetBlock(nn.Module):
    dim: int = 32
    groups: int = 8

    @nn.compact
    def __call__(self, inputs, time_embed=None):
        x = Block(self.dim, self.groups)(inputs)
        if time_embed is not None:
            time_embed = nn.silu(time_embed)
            time_embed = nn.Dense(self.dim)(time_embed)
            x = jnp.expand_dims(jnp.expand_dims(time_embed, 1), 1) + x
        x = Block(self.dim, self.groups)(x)
        res_conv = nn.Conv(self.dim, (1, 1), padding='SAME')(inputs)
        return x + res_conv


class MNISTUNet(nn.Module):
    dim: int = 8
    dim_scale_factor: tuple = (1, 2, 4)
    num_groups: int = 4

    @nn.compact
    def __call__(self, x, t):
        x = jnp.reshape(x, (-1, 28, 28, 1))
        t = jnp.reshape(t, (-1,))
        nchannels = x.shape[-1]
        x = nn.Conv(self.dim // 3 * 2, (7, 7), padding=((3, 3), (3, 3)))(x)
        time_emb = TimeEmbedding(self.dim)(t)

        dims = [self.dim * i for i in self.dim_scale_factor]
        pre_downsampling = []

        # Downsampling phase
        for index, dim in enumerate(dims):
            x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            att = Attention(dim)(x)
            norm = nn.GroupNorm(self.num_groups)(att)
            x = norm + x
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if index != len(dims) - 1:
                x = nn.Conv(dim, (4, 4), (2, 2))(x)

        # Middle block
        x = ResnetBlock(dims[-1], self.num_groups)(x, time_emb)
        att = Attention(dim)(x)
        norm = nn.GroupNorm(self.num_groups)(att)
        x = norm + x
        x = ResnetBlock(dims[-1], self.num_groups)(x, time_emb)

        # Upsampling phase
        for index, dim in enumerate(reversed(dims)):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            att = Attention(dim)(x)
            norm = nn.GroupNorm(self.num_groups)(att)
            x = norm + x
            if index != len(dims) - 1:
                x = nn.ConvTranspose(dim, (4, 4), (2, 2))(x)

        # Final ResNet block and output convolutional layer
        x = ResnetBlock(dim, self.num_groups)(x, time_emb)
        x = nn.Conv(nchannels, (1, 1), padding='SAME')(x)
        return jnp.squeeze(jnp.reshape(x, (x.shape[0], -1)))
