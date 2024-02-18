import jax
import jax.numpy as jnp
import itertools
from .base import Dataset
from fbs.typings import JKey, Array, JArray
from typing import Tuple


class MNIST(Dataset):
    """
    MNIST dataset.

    Data `x` has shape (n, 28, 28)
    """

    def __init__(self,
                 key: JKey,
                 data_path: str,
                 task: str = 'deconv-10',
                 test: bool = False):
        data_dict = jnp.load(data_path)
        self.task = task

        if test:
            self.n = 10000
            xs = data_dict['X_test']
            xs = jax.random.permutation(key, xs, axis=0)
        else:
            self.n = 60000
            xs = data_dict['X']
            xs = jax.random.permutation(key, xs, axis=0)

        self.xs = self.standardise(xs)

    @staticmethod
    def standardise(array: Array) -> JArray:
        return array

    @staticmethod
    def paint(key, img: Array, rectangle_size: int = 15) -> JArray:
        """Paint the image with a random rectangle.
        """

        def gen_mask(i, j):
            mask = jnp.ones((1, 28, 28, 1))
            return mask.at[:, i:i + rectangle_size, j: j + rectangle_size, :].set(0.)

        masks = jnp.concatenate([gen_mask(i, j) for (i, j) in itertools.product((0, 5, 10, 12), (0, 5, 10, 12))],
                                axis=0)
        return img * jax.random.choice(key, masks)

    @staticmethod
    def downsample(key: JArray, img: Array) -> Array:
        img = jnp.reshape(img, (1, 28, 28))

        kernel_sizes = jnp.array([2, 4, 8, 16])
        kernel_size = jax.random.choice(key, kernel_sizes)
        kernel = jnp.ones((kernel_size, kernel_size, 1, 1))

        corrupted_img = jax.lax.conv_general_dilated(img, kernel, (1, 1), 'SAME',
                                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0, :, :, 0]
        return (corrupted_img - jnp.min(corrupted_img)) / (jnp.max(corrupted_img) - jnp.min(corrupted_img))

    @staticmethod
    def conv(key: JKey, img: Array, kernel_size: int = 10) -> JArray:
        """Corrupt the image with a random Gaussian blur.

        Parameters
        ----------
        key : JKey
            Random key.
        img : Array (h, w)
            Image to corrupt.
        kernel_size

        Returns
        -------
        Array (h, w)
            Convoluted image.
        """
        img = jnp.reshape(img, (1, 28, 28))

        variance = jax.random.uniform(key, minval=0.1, maxval=5.)  # it's not conjugate yeah I know...
        z_ = jnp.dstack(jnp.meshgrid(jnp.linspace(-1, 1, kernel_size), jnp.linspace(-1, 1, kernel_size)))
        kernel = jnp.prod(jnp.exp(-z_ ** 2 / variance), axis=-1).reshape(kernel_size, kernel_size, 1, 1)
        corrupted_img = jax.lax.conv_general_dilated(img, kernel, (1, 1), 'SAME',
                                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0, :, :, 0]
        return (corrupted_img - jnp.min(corrupted_img)) / (jnp.max(corrupted_img) - jnp.min(corrupted_img))

    def corrupt(self, key: JKey, img: JArray) -> JArray:
        if 'inpaint' in self.task:
            rec_size = int(self.task.split('-')[-1])
            return self.paint(key, img, rec_size)
        elif 'deconv' in self.task:
            kernel_size = int(self.task.split('-')[-1])
            return self.conv(key, img, kernel_size)
        else:
            raise NotImplementedError('Not implemented.')

    def sampler(self, key: JKey) -> Tuple[JArray, JArray]:
        """Sample a pair of images from the dataset.

        Parameters
        ----------
        key : JKey
            Random key.

        Returns
        -------
        JArray (28, 28), JArray (28, 28)
            A pair of clean and corrupted images.
        """
        key_choice, key_corrupt = jax.random.split(key)
        x = self.xs[jax.random.choice(key_choice, self.n)]
        y = self.corrupt(key_corrupt, x)
        return x, y

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> Tuple[JArray, JArray]:
        if perm_inds is None:
            perm_inds = self.perm_inds
        inds = perm_inds[i]

        xs = self.xs[inds]
        keys = jax.random.split(key, num=inds.shape[0])
        ys = jax.vmap(self.corrupt, in_axes=[0, 0])(keys, xs)
        return xs, ys

    @staticmethod
    def concat(x: JArray, y: JArray) -> JArray:
        return jnp.concatenate([jnp.expand_dims(x, -1), jnp.expand_dims(y, -1)], axis=-1)

    @staticmethod
    def unpack(xy: JArray) -> Tuple[JArray, JArray]:
        return xy[..., 0], xy[..., 1]


class CIFAR10(Dataset):
    """
    CIFAR10 dataset.

    Data `x` has shape (n, 28, 28)
    """

    def __init__(self,
                 key: JKey,
                 data_path: str,
                 task: str = 'deconv-10',
                 test: bool = False):
        data_dict = jnp.load(data_path)
        self.task = task

        if test:
            self.n = 10000
            xs = data_dict['test_data']
            xs = jax.random.permutation(key, xs, axis=0)
            self.xs = jnp.reshape(xs, (10000, 32, 32, 3))
        else:
            self.n = 50000
            xs = data_dict['train_data']
            xs = jax.random.permutation(key, xs, axis=0)
            self.xs = jnp.reshape(xs, (50000, 32, 32, 3))

        self.xs = self.standardise(xs)
        self.image_shape = (32, 32, 3)

    @staticmethod
    def standardise(array: Array) -> JArray:
        return array

    def downsample(self, key: JArray, img: Array) -> Array:
        img = jnp.reshape(img, (1, *self.image_shape))

        def down(ratio):
            return jax.image.resize(jax.image.resize(img, (1, int(32 / ratio), int(32 / ratio), 3), 'nearest'),
                                    (1, 32, 32, 3), 'nearest')

        imgs = jnp.concatenate([down(i) for i in (4, 8)], axis=0)
        return jax.random.choice(key, imgs)

    def conv(self, key: JKey, img: Array, kernel_size: int = 10) -> JArray:
        """Corrupt the image with a random Gaussian blur.

        Parameters
        ----------
        key : JKey
            Random key.
        img : Array (h, w, c)
            Image to corrupt.
        kernel_size

        Returns
        -------
        Array (h, w, c)
            Convoluted image.
        """
        nchannels = self.image_shape[-1]
        img = jnp.reshape(img, (1, *self.image_shape))

        variance = jax.random.uniform(key, minval=0.1, maxval=5.)  # it's not conjugate yeah I know...
        z_ = jnp.dstack(jnp.meshgrid(jnp.linspace(-1, 1, kernel_size), jnp.linspace(-1, 1, kernel_size)))
        kernel = jnp.broadcast_to(jnp.prod(jnp.exp(-z_ ** 2 / variance), axis=-1),
                                  (nchannels, nchannels, kernel_size, kernel_size))
        corrupted_img = jax.lax.conv_general_dilated(img, kernel, (1, 1), 'SAME',
                                                     dimension_numbers=('NHWC', 'IOHW', 'NHWC'))[0, :, :]
        return normalise_rgb(corrupted_img)

    def corrupt(self, key: JKey, img: JArray) -> JArray:
        if 'inpaint' in self.task:
            raise NotImplementedError('Inpainting is not implemented.')
        elif 'deconv' in self.task:
            kernel_size = int(self.task.split('-')[-1])
            return self.conv(key, img, kernel_size)
        elif 'supr' in self.task:
            return self.downsample(key, img)
        else:
            raise NotImplementedError('Not implemented.')

    def sampler(self, key: JKey) -> Tuple[JArray, JArray]:
        """Sample a pair of images from the dataset.

        Parameters
        ----------
        key : JKey
            Random key.

        Returns
        -------
        JArray (28, 28, c), JArray (28, 28, c)
            A pair of clean and corrupted images.
        """
        key_choice, key_corrupt = jax.random.split(key)
        x = self.xs[jax.random.choice(key_choice, self.n)]
        y = self.corrupt(key_corrupt, x)
        return x, y

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> Tuple[JArray, JArray]:
        if perm_inds is None:
            perm_inds = self.perm_inds
        inds = perm_inds[i]

        xs = self.xs[inds]
        keys = jax.random.split(key, num=inds.shape[0])
        ys = jax.vmap(self.corrupt, in_axes=[0, 0])(keys, xs)
        return xs, ys

    @staticmethod
    def concat(x: JArray, y: JArray, expand: bool = False) -> JArray:
        if expand:
            return jnp.concatenate([jnp.expand_dims(x, -1), jnp.expand_dims(y, -1)], axis=-1)
        else:
            return jnp.concatenate([x, y], axis=-1)

    @staticmethod
    def unpack(xy: JArray) -> Tuple[JArray, JArray]:
        return xy[..., :3], xy[..., 3:]


def normalise_rgb(img: JArray) -> JArray:
    mins = jnp.min(img, axis=[-2, -3], keepdims=True)
    maxs = jnp.max(img, axis=[-2, -3], keepdims=True)
    return (img - mins) / (maxs - mins)
