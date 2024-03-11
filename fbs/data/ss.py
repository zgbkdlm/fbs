import math
import jax.numpy as jnp
import jax.random
from fbs.data.base import Dataset
from fbs.typings import JArray, JKey
from typing import Tuple


class StochasticVolatility(Dataset):
    r"""
    ...
    """

    def __init__(self, n: int = 10, psi: float = 1., xi: float = 1.):
        self.n = n
        self.psi = psi
        self.m = jnp.array([0., 0.])
        self.cov = jnp.array([[2., 0.],
                              [0., 1.]])
        self.cov_is_diag = True
        self.xi = xi

    def sampler(self, key: JKey, batch_size: int) -> Tuple[JArray, JArray]:
        key, subkey = jax.random.split(key)
        xs = self.m + jax.random.normal(subkey, (batch_size, 2)) @ jnp.linalg.cholesky(self.cov)

        key, subkey = jax.random.split(key)
        ys = (jax.vmap(self.emission, in_axes=[0, None])(xs, self.psi)
              + math.sqrt(self.xi) * jax.random.normal(subkey, (batch_size,)))
        return xs, ys

    @staticmethod
    def unpack(xy):
        return xy[..., :2], xy[..., -1]
