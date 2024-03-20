import jax
import jax.numpy as jnp
import itertools
from .base import Dataset
from fbs.typings import JKey, Array, JArray
from functools import partial
from typing import Tuple, NamedTuple, Union


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
        if self.task == 'none':
            return xs, None
        else:
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


class InpaintingMask(NamedTuple):
    """The mask that selects the hollow part.
    """
    width: int
    height: int
    shift: JArray
    unobs_inds_ravelled: JArray
    obs_inds_ravelled: JArray


class SRMask(NamedTuple):
    rate: int
    unobs_inds_ravelled: JArray
    obs_inds_ravelled: JArray


class ImageRestore(Dataset):
    image_shape: Tuple[int, int, int]
    task: str
    unobs_shape: Tuple[int, int]

    def __init__(self, task: str):
        w, h, c = self.image_shape
        s = int(self.task.split('-')[-1])
        if 'inpaint' in task:
            self.unobs_shape = (s ** 2, c)
        elif 'supr' in task:
            self.unobs_shape = (int(w * h * (s ** 2 - 1) / s ** 2), c)
        else:
            raise ValueError(f'Unknown task {task}.')

    @staticmethod
    def standardise(array: Array) -> JArray:
        return array

    def enumerate_subset(self, i: int, perm_inds=None, key=None) -> JArray:
        if perm_inds is None:
            perm_inds = self.perm_inds
        inds = perm_inds[i]
        return self.xs[inds]

    def _gen_supr_mask(self, key: JKey, rate: int) -> SRMask:
        img_w, img_h = self.image_shape[:2]
        nblocks = int(img_w * img_h / rate ** 2)
        shifts = jax.random.randint(key, (nblocks, 2), 0, rate)

        inds_w, inds_h = [i for i in range(0, img_w, rate)], [i for i in range(0, img_h, rate)]
        inds_wa, inds_ha = [i for i in range(img_w)], [i for i in range(img_h)]
        block_inds = jnp.asarray(list(itertools.product(inds_w, inds_h)))
        all_inds = jnp.asarray(list(itertools.product(inds_wa, inds_ha)))

        block_inds_ravelled = jnp.ravel_multi_index([block_inds[:, 0] + shifts[:, 0], block_inds[:, 1] + shifts[:, 1]],
                                                    (img_w, img_h), mode='clip')
        all_inds_ravelled = jnp.ravel_multi_index([all_inds[:, 0], all_inds[:, 1]], (img_w, img_h), mode='clip')
        unobs_inds_ravelled = jnp.setdiff1d(all_inds_ravelled, block_inds_ravelled, assume_unique=True,
                                            size=img_w * img_h - nblocks)
        return SRMask(rate, unobs_inds_ravelled=unobs_inds_ravelled, obs_inds_ravelled=block_inds_ravelled)

    def _gen_inpaint_mask(self, key: JKey, width: int, height: int) -> InpaintingMask:
        """Note that this might not be jitted due to `setdiff1d`.
        """
        img_w, img_h = self.image_shape[:2]
        width, height = min(width, img_w), min(height, img_h)
        inds_w, inds_h = [i for i in range(width)], [i for i in range(height)]
        inds_wa, inds_ha = [i for i in range(img_w)], [i for i in range(img_h)]
        rect_inds = jnp.asarray(list(itertools.product(inds_w, inds_h)))
        all_inds = jnp.asarray(list(itertools.product(inds_wa, inds_ha)))

        max_shift = min(img_w, img_h) - max(width, height)
        shift = jax.random.randint(key, (), 0, max_shift)
        rect_inds_ravelled = jnp.ravel_multi_index([rect_inds[:, 0] + shift, rect_inds[:, 1] + shift],
                                                   (img_w, img_h), mode='clip')
        all_inds_ravelled = jnp.ravel_multi_index([all_inds[:, 0], all_inds[:, 1]], (img_w, img_h), mode='clip')
        obs_inds_ravelled = jnp.setdiff1d(all_inds_ravelled, rect_inds_ravelled, assume_unique=True,
                                          size=img_w * img_h - width * height)
        return InpaintingMask(width, height, shift,
                              unobs_inds_ravelled=rect_inds_ravelled, obs_inds_ravelled=obs_inds_ravelled)

    def gen_mask(self, key: JKey):
        if 'inpaint' in self.task:
            s = int(self.task.split('-')[-1])
            return self._gen_inpaint_mask(key, s, s)
        elif 'supr' in self.task:
            rate = int(self.task.split('-')[-1])
            return self._gen_supr_mask(key, rate)
        else:
            raise ValueError(f'Unknown task {self.task}.')

    def sampler(self, key: JKey) -> Tuple[JArray, JArray, Union[InpaintingMask, SRMask]]:
        """

        Parameters
        ----------
        key

        Returns
        -------
        JArray (w, h, c), JArray (q, c), InpaintingMask
            True image, observed part of the image, and the inpainting mask.
        """
        key_choice, key_corrupt = jax.random.split(key)
        x = self.xs[jax.random.choice(key_choice, self.n)]

        mask = self.gen_mask(key_corrupt)
        _, y = self.unpack(x, mask)
        return x, y, mask

    def unpack(self, xy: JArray, mask: InpaintingMask) -> Tuple[JArray, JArray]:
        """Decompose an image into two parts, viz., the painted and original parts.

        Parameters
        ----------
        xy : JArray (..., h, w, c)
            The image to be decomposed.
        mask : InpaintingMask

        Returns
        -------
        JArray (..., p, c), JArray (..., q, c)
            The painted and original parts.
        """
        img_w, img_h, img_c = self.image_shape
        rect_inds_ravelled, obs_inds_ravelled = mask.unobs_inds_ravelled, mask.obs_inds_ravelled

        xy_ravelled = jnp.reshape(xy, (*xy.shape[:-3], img_w * img_h, img_c))
        x = xy_ravelled[..., rect_inds_ravelled, :]
        y = xy_ravelled[..., obs_inds_ravelled, :]
        return x, y

    def concat(self, x: JArray, y: JArray, mask: InpaintingMask) -> JArray:
        """The reverse operation of `unpack2`."""
        img_w, img_h, img_c = self.image_shape
        unobs_inds_ravelled, obs_inds_ravelled = mask.unobs_inds_ravelled, mask.obs_inds_ravelled

        img = jnp.zeros((*x.shape[:-3], img_w * img_h, img_c))
        img = img.at[..., unobs_inds_ravelled, :].set(x)
        img = img.at[..., obs_inds_ravelled, :].set(y)
        return img.reshape(*img.shape[:-3], img_w, img_h, img_c)


class MNISTRestore(ImageRestore):
    def __init__(self,
                 key: JKey,
                 data_path: str,
                 task: str = 'inpaint-15',
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
        super().__init__(task)


class CelebAHQRestore(ImageRestore):

    def __init__(self,
                 key: JKey,
                 data_path: str,
                 resolution: int = 64,
                 task: str = 'supr-4',
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
        super().__init__(task)


def normalise(img: JArray, method: str = 'clip') -> JArray:
    if method == 'clip':
        img = jnp.where(img < 0, 0., img)
        img = jnp.where(img > 1, 1., img)
        return img
    else:
        mins = jnp.min(img, axis=[-2, -3], keepdims=True)
        maxs = jnp.max(img, axis=[-2, -3], keepdims=True)
        return (img - mins) / (maxs - mins)
