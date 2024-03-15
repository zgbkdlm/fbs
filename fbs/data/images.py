import jax
import jax.numpy as jnp
import itertools
from .base import Dataset
from fbs.typings import JKey, Array, JArray
from functools import partial
from typing import Tuple, Sequence


class Image(Dataset):
    image_shape: Tuple[int, int, int]
    task: str

    @staticmethod
    def standardise(array: Array) -> JArray:
        return array

    def downsample(self, key: JArray, img: Array) -> Array:
        ratios = (4, 8)
        w, h, c = self.image_shape

        def down(ratio):
            return jax.image.resize(jax.image.resize(img, (int(w / ratio), int(h / ratio), c), 'nearest'),
                                    (w, h, c), 'nearest')[None, ...]

        imgs = jnp.concatenate([down(ratio) for ratio in ratios], axis=0)
        return jax.random.choice(key, imgs)

    def conv(self, key: JKey, img: Array, kernel_size: int = 15) -> JArray:
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
        c = self.image_shape[-1]
        img = jnp.reshape(img, (1, *self.image_shape))

        variance = jax.random.uniform(key, minval=0.1, maxval=5.)  # it's not conjugate yeah I know...
        z_ = jnp.dstack(jnp.meshgrid(jnp.linspace(-1, 1, kernel_size), jnp.linspace(-1, 1, kernel_size)))
        kernel = jnp.broadcast_to(jnp.prod(jnp.exp(-z_ ** 2 / variance), axis=-1),
                                  (c, c, kernel_size, kernel_size))
        corrupted_img = jax.lax.conv_general_dilated(img, kernel, (1, 1), 'SAME',
                                                     dimension_numbers=('NHWC', 'IOHW', 'NHWC'))[0]
        return normalise(corrupted_img, method='norm')

    def paint(self, key, img: Array, rectangle_size: int = 15) -> JArray:
        """Paint the image with a random rectangle.
        """
        b = 3
        h, w = self.image_shape[:2]
        stride_h, stride_w = (h - rectangle_size) / b, (w - rectangle_size) / b
        hs, ws = [int(stride_h * i) for i in range(b + 1)], [int(stride_w * i) for i in range(b + 1)]

        def gen_mask(i, j):
            mask = jnp.ones(self.image_shape)
            return mask.at[i:i + rectangle_size, j: j + rectangle_size, :].set(0.)

        masks = jnp.concatenate([gen_mask(i, j)[None, ...] for (i, j) in itertools.product(hs, ws)],
                                axis=0)
        return img * jax.random.choice(key, masks)

    def corrupt(self, key: JKey, img: JArray) -> JArray:
        if 'inpaint' in self.task:
            rectangle_size = int(self.task.split('-')[-1])
            return self.paint(key, img, rectangle_size=rectangle_size)
        elif 'deconv' in self.task:
            kernel_size = int(self.task.split('-')[-1])
            return self.conv(key, img, kernel_size)
        elif 'supr' in self.task:
            return self.downsample(key, img)
        else:
            raise ValueError(f'Unknown task {self.task}.')

    def sampler(self, key: JKey) -> Tuple[JArray, JArray]:
        """Sample a pair of images from the dataset.

        Parameters
        ----------
        key : JKey
            Random key.

        Returns
        -------
        JArray (w, h, c), JArray (w, h, c)
            A pair of clean and corrupted images.
        """
        key_choice, key_corrupt = jax.random.split(key)
        x = self.xs[jax.random.choice(key_choice, self.n)]
        y = self.corrupt(key_corrupt, x)
        return x, y

    @partial(jax.jit, static_argnums=0)
    def _enumerate_jit(self, inds, key):
        xs = self.xs[inds]
        keys = jax.random.split(key, num=inds.shape[0])
        ys = jax.vmap(self.corrupt, in_axes=[0, 0])(keys, xs)
        return xs, ys

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> Tuple[JArray, JArray]:
        if perm_inds is None:
            perm_inds = self.perm_inds
        inds = perm_inds[i]
        return self._enumerate_jit(inds, key)

    @staticmethod
    def concat(x: JArray, y: JArray, expand: bool = False) -> JArray:
        if expand:
            return jnp.concatenate([jnp.expand_dims(x, -1), jnp.expand_dims(y, -1)], axis=-1)
        else:
            return jnp.concatenate([x, y], axis=-1)

    def unpack(self, xy: JArray) -> Tuple[JArray, JArray]:
        c = self.image_shape[-1]
        return xy[..., :c], xy[..., c:]


class MNIST(Image):
    """
    MNIST dataset.

    Data `x` has shape (n, 28, 28)
    """

    def __init__(self,
                 key: JKey,
                 data_path: str,
                 task: str = 'deconv-15',
                 test: bool = False):
        data_dict = jnp.load(data_path)
        self.task = task

        if test:
            self.n = 10000
            xs = data_dict['X_test']
            xs = jax.random.permutation(key, xs, axis=0)
            xs = jnp.reshape(xs, (10000, 28, 28, 1))
        else:
            self.n = 60000
            xs = data_dict['X']
            xs = jax.random.permutation(key, xs, axis=0)
            xs = jnp.reshape(xs, (60000, 28, 28, 1))

        self.xs = self.standardise(xs).astype('float32')
        self.image_shape = (28, 28, 1)


class CIFAR10(Image):
    """
    CIFAR10 dataset.

    Data `x` has shape (n, 32, 32, 3)
    """

    def __init__(self,
                 key: JKey,
                 data_path: str,
                 task: str = 'supr',
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


class CelebAHQ(Image):
    def __init__(self,
                 key: JKey,
                 data_path: str,
                 resolution: int = 64,
                 task: str = 'supr',
                 test: bool = False):
        self.task = task
        data = jnp.load(data_path)
        data = jax.random.permutation(key, data, axis=0)
        data = self.standardise(data)

        if test:
            self.n = 1000
            self.xs = data[:1000]
        else:
            self.n = 29000
            self.xs = data[1000:]

        self.image_shape = (resolution, resolution, 3)


class MNISTInpaint(MNIST):
    @staticmethod
    def concat(x: JArray, y: JArray) -> JArray:
        return jnp.concatenate([x, y], axis=-2)

    def unpack(self, xy: JArray) -> Tuple[JArray, JArray]:
        split = 14
        return xy[..., :, :split, :], xy[..., :, split:, :]


class CelebAHQInpaint(CelebAHQ):
    @staticmethod
    def concat(x: JArray, y: JArray) -> JArray:
        return jnp.concatenate([x, y], axis=-2)

    def unpack(self, xy: JArray) -> Tuple[JArray, JArray]:
        split = 14
        return xy[..., :, :split, :], xy[..., :, split:, :]

    def unpack2(self, xy: JArray) -> Tuple[JArray, JArray]:
        """Decompose an image into two parts, viz., the painted and original parts.

        Parameters
        ----------
        xy : JArray (..., h, w, c)
            The image to be decomposed.

        Returns
        -------
        JArray (..., p, c), JArray (..., q, c)
            The painted and original parts.
        """
        resolution = self.image_shape[0]
        width, height = 15, 15
        inds = [i for i in range(width)]
        mask = jnp.asarray(list(itertools.product(inds, inds))).T
        shift = 20  # Random
        mask_ravelled = jnp.ravel_multi_index(mask + shift, (resolution, resolution))

        xy_ravelled = jnp.reshape(xy, (*xy.shape[:-3], resolution ** 2, 3))
        x = xy_ravelled[..., mask_ravelled, :]
        y_masks = jnp.ones(resolution ** 2, dtype=bool)
        y_masks = y_masks.at[mask_ravelled].set(False)
        y = xy_ravelled[..., y_masks, :]
        return x, y

    def concat2(self, x: JArray, y: JArray) -> JArray:
        """The reverse operation of `unpack2`."""
        resolution = self.image_shape[0]
        img = jnp.zeros((x.shape[:-2], resolution ** 2, 3))
        img = img.at[..., mask_x, :].set(x)
        img = img.at[..., mask_y, :].set(y)
        return img


def normalise(img: JArray, method: str = 'clip') -> JArray:
    if method == 'clip':
        img = jnp.where(img < 0, 0., img)
        img = jnp.where(img > 1, 1., img)
        return img
    else:
        mins = jnp.min(img, axis=[-2, -3], keepdims=True)
        maxs = jnp.max(img, axis=[-2, -3], keepdims=True)
        return (img - mins) / (maxs - mins)
