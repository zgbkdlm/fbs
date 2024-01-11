import jax
import jax.numpy as jnp
import numpy as np
from fbs.typings import Array, JArray, JKey
from abc import ABCMeta
from typing import List, Tuple


class DataSet(metaclass=ABCMeta):
    """
    An abstract class for datasets.

    There are two types of datasets. One has access to the sampler p(x, y), while the another only has access to a
    fixed size of data samples. The former uses `generate`, and the latter uses `enumerate` for training.
    """
    n: int
    xs: Array
    ys: Array
    perm_inds: List[JArray]

    @staticmethod
    def reshape(x: Array) -> JArray:
        if x.ndim == 0:
            return jnp.reshape(x, (1, 1))
        elif x.ndim == 1:
            return jnp.reshape(x, (-1, 1))
        else:
            return x

    @staticmethod
    def standardise(array: Array) -> JArray:
        """Standardise for array of shape (n, d).
        """
        return (array - jnp.mean(array, axis=0)) / jnp.std(array, axis=0)

    def draw_subset(self, key: JKey, batch_size: int) -> Tuple[JArray, JArray]:
        inds = jax.random.choice(key, jnp.arange(self.n), (batch_size,), replace=False)
        return self.reshape(self.xs[inds, :]), self.reshape(self.ys[inds, :])

    def init_enumeration(self, key: JKey, batch_size: int) -> List[JArray]:
        """Randomly split the data into `n / batch_size` chunks. If the divisor is not an integer, then use // which
        truncates the training data.
        """
        n_chunks = self.n // batch_size
        perm_inds = jnp.array_split(jax.random.choice(key,
                                                      jnp.arange(batch_size * n_chunks), (batch_size * n_chunks,),
                                                      replace=False),
                                    n_chunks)
        self.perm_inds = perm_inds
        return perm_inds

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> Tuple[JArray, JArray]:
        """Enumerate all the randomly split chunks of data for i = 0, 1, ...
        """
        inds = self.perm_inds[i]
        return self.reshape(self.xs[inds, :]), self.reshape(self.ys[inds, :])

    def sampler(self, key: JKey) -> Tuple[JArray, JArray]:
        raise NotImplementedError('Not implemented.')
