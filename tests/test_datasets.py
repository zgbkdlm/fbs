import jax.numpy as jnp
import jax.random
import numpy.testing as npt
from fbs.data.base import DataSet


class TestDatasetClass:
    def test_enumeration(self):
        dummyclass = DataSet()

        data_size = 100
        dummyclass.n = data_size
        key = jax.random.PRNGKey(666)
        dummyclass.xs, dummyclass.ys = jax.random.normal(key, (2, 100, 3))
        batch_size = 4
        key, _ = jax.random.split(key)
        dummyclass.init_enumeration(key, batch_size)

        npt.assert_array_equal(jnp.sort(jnp.concatenate(dummyclass.perm_inds)), jnp.arange(data_size))

        xss = []
        yss = []
        for i in range(int(data_size / batch_size)):
            xs, ys = dummyclass.enumerate_subset(i)
            xss.append(xs)
            yss.append(ys)

        # Generated data should be a random permutation of the original
        npt.assert_array_equal(jnp.sort(jnp.concatenate(xss).ravel()), jnp.sort(dummyclass.xs.ravel()))
        npt.assert_array_equal(jnp.sort(jnp.concatenate(yss).ravel()), jnp.sort(dummyclass.ys.ravel()))
