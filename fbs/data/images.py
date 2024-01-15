import jax
import jax.numpy as jnp
from .base import DataSet
from fbs.typings import JKey, Array, JArray
from typing import Tuple


class MNIST(DataSet):

    def __init__(self, key: JKey, data_path: str, task: str = 'inpainting'):
        data_dict = jnp.load(data_path)
        self.n = 70000
        self.task = task

        xs = jnp.concatenate([data_dict['X'], data_dict['X_test']], axis=0)
        xs = jax.random.permutation(key, jnp.reshape(xs, (70000, 784)), axis=0)

        self.xs = self.standardise(xs)

    @staticmethod
    def standardise(array: Array) -> JArray:
        return array

    @staticmethod
    def paint(key, img: Array, paint_val: float = 0.) -> JArray:
        """Paint the image with a random rectangle.
        """
        img = jnp.reshape(img, (28, 28))
        start_ind_x, start_ind_y = jax.random.randint(key, (2,), minval=5, maxval=10)

        key, subkey = jax.random.split(key)
        width, height = jax.random.randint(subkey, (2,), minval=10, maxval=17)

        img = img.at[start_ind_x:start_ind_x + width, start_ind_y:start_ind_y + height].set(paint_val)  # Not doable

    @staticmethod
    def conv(key: JKey, img: Array) -> JArray:
        kernel = jnp.ones((6, 6)) * jax.random.uniform(key, minval=0.2, maxval=0.8) / 12
        return jax.scipy.signal.convolve(img, kernel, mode='same')

    def corrupt(self, key: JKey, img: JArray) -> JArray:
        if self.task == 'inpainting':
            raise NotImplementedError('Not implemented.')
        elif self.task == 'deconv':
            return self.conv(key, img)
        else:
            raise ValueError('Unknown task.')

    def sampler(self, key: JKey) -> Tuple[JArray, JArray]:
        key_choice, key_corrupt = jax.random.split(key)
        x = self.xs[jax.random.choice(key_choice, self.n)]
        y = self.corrupt(key_corrupt, x.reshape(28, 28)).reshape(784)
        return x, y

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> Tuple[JArray, JArray]:
        if perm_inds is None:
            perm_inds = self.perm_inds
        inds = perm_inds[i]

        xs = self.xs[inds, :]
        keys = jax.random.split(key, num=inds.shape[0])
        ys = jax.vmap(self.corrupt,
                      in_axes=[0, 0])(keys, xs.reshape(inds.shape[0], 28, 28)).reshape(inds.shape[0], 784)
        return xs, ys
