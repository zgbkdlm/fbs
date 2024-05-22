import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.utils import kl, bures_dist

jax.config.update("jax_enable_x64", True)


def test_kl_bures():
    m0, m1 = jnp.ones((2, 10))
    cov0, cov1 = jnp.eye(10), jnp.eye(10)
    npt.assert_allclose(0., kl(m0, cov0, m1, cov1))
    npt.assert_allclose(0., bures_dist(m0, cov0, m1, cov1))
