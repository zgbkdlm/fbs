import numpy.testing as npt
import jax.numpy as jnp
import torch
from fbs.nn.utils import PixelShuffle


def test_pixel_shuffle():
    torch.manual_seed(666)
    img_torch = torch.randn(3, 4, 2, 2)
    out_torch = torch.nn.PixelShuffle(2)(img_torch)
    out_torch = jnp.transpose(jnp.asarray(out_torch), [0, 2, 3, 1])

    img_jax = jnp.transpose(jnp.asarray(img_torch), [0, 2, 3, 1])
    out_jax = PixelShuffle(2)(img_jax)

    npt.assert_allclose(out_torch, out_jax, rtol=1e-5, atol=1e-5)
