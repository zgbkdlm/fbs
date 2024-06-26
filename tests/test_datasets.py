import jax.numpy as jnp
import jax.random
import numpy.testing as npt
import scipy
import pytest
from fbs.data.base import Dataset
from fbs.data import Crescent, ImageRestore
from fbs.sdes import euler_maruyama


class DummyClass(Dataset):
    pass


class TestDatasetClass:
    def test_enumeration(self):
        dummyclass = DummyClass()

        data_size = 100
        dummyclass.n = data_size
        key = jax.random.PRNGKey(666)
        dummyclass.xs, dummyclass.ys = jax.random.normal(key, (2, 100, 3))
        batch_size = 4
        key, _ = jax.random.split(key)
        dummyclass.init_enumeration(key, batch_size)

        npt.assert_array_equal(jnp.sort(jnp.concatenate(dummyclass.perm_inds)), jnp.arange(data_size))

        xss = []
        for i in range(int(data_size / batch_size)):
            xs = dummyclass.enumerate_subset(i)
            xss.append(xs)

        # Generated data should be a random permutation of the original
        npt.assert_array_equal(jnp.sort(jnp.concatenate(xss).ravel()), jnp.sort(dummyclass.xs.ravel()))


class TestCrescent:
    def test_score(self):
        crescent = Crescent()

        T = 5
        nsteps = 100
        ts = jnp.linspace(0, T, nsteps + 1)

        def drift(xy, t):
            return 0.5 * crescent.score(xy)

        def dispersion(t):
            return 1.

        def sampler_xy(key_):
            x_, y_ = crescent.sampler(key_, 1)
            return jnp.hstack([x_[0], y_[0]])

        def fwd_simulator(key_):
            key1, key2 = jax.random.split(key_, num=2)
            xy0 = sampler_xy(key1)
            return euler_maruyama(key2, xy0, ts, drift, dispersion, return_path=True)

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, num=10000)
        paths = jax.vmap(fwd_simulator)(keys)
        true_samples = paths[:, 0, :]

        test_steps = [5, 100, 400, -1]
        for step in test_steps:
            langevin_samples = paths[:, step, :]
            for dim in range(3):
                npt.assert_allclose(scipy.stats.wasserstein_distance(true_samples[:, dim], langevin_samples[:, dim]),
                                    0., atol=2e-1)


class TestImgs:

    @pytest.mark.parametrize('task', ['inpainting-8', 'supr-4'])
    def test_concat_unpack(self, task):
        img_shape = (32, 32, 3)
        key = jax.random.PRNGKey(666)

        dataset = ImageRestore(task=task, image_shape=img_shape, sr_random=True)

        key, subkey = jax.random.split(key)
        true_img = jax.random.uniform(subkey, (4, *img_shape))

        key, subkey = jax.random.split(key)
        mask = dataset.gen_mask(subkey)
        x, y = dataset.unpack(true_img, mask)
        xy = dataset.concat(x, y, mask)

        npt.assert_array_equal(true_img, xy)
