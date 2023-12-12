"""Test file for resampling methods."""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from .resamplings import multinomial, systematic, killing

RESAMPLINGS = {"multinomial": multinomial, "systematic": systematic, "killing": killing}


@pytest.mark.parametrize("resampling", ["multinomial", "systematic", "killing"])
@pytest.mark.parametrize("seed", [42, 666, 1234])
def test_unconditional(resampling, seed):
    JAX_KEY = jax.random.PRNGKey(seed)
    resampling_fn = lambda k, w: RESAMPLINGS[resampling](k, w, conditional=False)

    keys = jax.random.split(JAX_KEY, 100_000)

    weights = jnp.cos(jnp.linspace(0, 2 * jnp.pi, 1000)) + 1
    weights /= jnp.sum(weights)

    indices = jax.vmap(resampling_fn, in_axes=[0, None])(keys, weights)
    bincount = np.bincount(indices[:, 1:].ravel(), minlength=weights.shape[0])

    npt.assert_allclose(bincount / np.sum(bincount), weights, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("resampling", ["multinomial", "systematic", "killing"])
@pytest.mark.parametrize("seed", [42, 666])
@pytest.mark.parametrize("i, j", [(0, 0), (5, 7), (7, 1)])
def test_conditional_preserves_ij(resampling, seed, i, j):
    JAX_KEY = jax.random.PRNGKey(seed)
    resampling_fn = lambda k, w: RESAMPLINGS[resampling](k, w, i, j, conditional=True)

    keys = jax.random.split(JAX_KEY, 1_000)

    weights = jnp.cos(jnp.linspace(0, 2 * jnp.pi, 50)) + 1
    weights /= jnp.sum(weights)

    indices = jax.vmap(resampling_fn, in_axes=[0, None])(keys, weights)
    # bincount = np.bincount(indices[:, 1:].ravel(), minlength=weights.shape[0])

    npt.assert_allclose(indices[:, j], i, atol=1e-3)


@pytest.mark.parametrize("resampling", ["multinomial", "systematic", "killing"])
@pytest.mark.parametrize("seed", [42, 666])
@pytest.mark.parametrize("j", [0, 5, 50])
def test_conditional_bayes(resampling, seed, j):
    N = 100
    JAX_KEY = jax.random.PRNGKey(seed)
    resampling_fn = lambda k, w, i: RESAMPLINGS[resampling](k, w, i, j, conditional=True)

    keys = jax.random.split(JAX_KEY, 100_000)

    weights = jnp.cos(jnp.linspace(0, 2 * jnp.pi, N)) + 1
    weights /= jnp.sum(weights)

    def bayes_sample(key):
        key1, key2 = jax.random.split(key)
        i = jax.random.choice(key1, N, p=weights)
        return resampling_fn(key2, weights, i)

    indices = jax.vmap(bayes_sample)(keys)

    bincount = np.bincount(indices[:, 1:].ravel(), minlength=weights.shape[0])
    npt.assert_allclose(bincount / np.sum(bincount), weights, atol=1e-3, rtol=1e-3)