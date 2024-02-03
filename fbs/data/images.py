import jax
import jax.numpy as jnp
from .base import Dataset
from fbs.typings import JKey, Array, JArray
from typing import Tuple


class MNIST(Dataset):

    def __init__(self, key: JKey, data_path: str, task: str = 'inpainting', test: bool = False):
        data_dict = jnp.load(data_path)
        self.task = task

        if test:
            self.n = 10000
            xs = data_dict['X_test']
            xs = jax.random.permutation(key, jnp.reshape(xs, (10000, 784)), axis=0)
        else:
            self.n = 60000
            xs = data_dict['X']
            xs = jax.random.permutation(key, jnp.reshape(xs, (60000, 784)), axis=0)

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
        """Corrupt the image with a random Gaussian blur.

        Parameters
        ----------
        key : JKey
            Random key.
        img : Array (h, w)
            Image to corrupt.

        Returns
        -------
        Array (h, w)
            Convoluted image.
        """
        img = jnp.reshape(img, (1, 28, 28, 1))

        variance = jax.random.uniform(key, minval=0.5, maxval=2.)  # it's not conjugate yeah I know...
        z_ = jnp.dstack(jnp.meshgrid(jnp.linspace(-1, 1, 5), jnp.linspace(-1, 1, 5)))
        kernel = jnp.prod(jnp.exp(-z_ ** 2 / variance), axis=-1).reshape(5, 5, 1, 1)
        corrupted_img = jax.lax.conv_general_dilated(img, kernel, (1, 1), 'SAME',
                                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        return (corrupted_img - jnp.min(corrupted_img)) / (jnp.max(corrupted_img) - jnp.min(corrupted_img))

    def corrupt(self, key: JKey, img: JArray) -> JArray:
        if self.task == 'inpainting':
            raise NotImplementedError('Not implemented.')
        elif self.task == 'deconv':
            return self.conv(key, img)
        else:
            raise NotImplementedError('Not implemented.')

    def sampler(self, key: JKey) -> Tuple[JArray, JArray]:
        key_choice, key_corrupt = jax.random.split(key)
        x = self.xs[jax.random.choice(key_choice, self.n)]
        y = self.corrupt(key_corrupt, x.reshape(28, 28)).reshape(784)
        return x, y

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> Tuple[JArray, JArray]:
        if perm_inds is None:
            perm_inds = self.perm_inds
        inds = perm_inds[i]
        batch_size = inds.shape[0]

        xs = self.xs[inds, :]
        keys = jax.random.split(key, num=inds.shape[0])
        ys = jax.vmap(self.corrupt,
                      in_axes=[0, 0])(keys, xs.reshape(batch_size, 28, 28)).reshape(batch_size, 784)
        return xs, ys

    @staticmethod
    def concat(x: JArray, y: JArray) -> JArray:
        batch_shape = x.shape[:-1]
        x = jnp.reshape(x, (*batch_shape, 28, 28, 1))
        y = jnp.reshape(y, (*batch_shape, 28, 28, 1))
        return jnp.reshape(jnp.concatenate([x, y], axis=-1), (*batch_shape, 784 * 2))

    @staticmethod
    def unpack(xy: JArray) -> Tuple[JArray, JArray]:
        batch_shape = xy.shape[:-1]
        xy = jnp.reshape(xy, (*batch_shape, 28, 28, 2))
        x, y = jnp.split(xy, 2, axis=-1)
        return x, y

