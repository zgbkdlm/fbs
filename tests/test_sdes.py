import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.sdes import make_ou_sde

jax.config.update("jax_enable_x64", True)


def test_ou_simulators():
    a, b = -0.8, 2.1
    stationary_mean, stationary_var = 0., b ** 2 / (2 * -a)

    key = jax.random.PRNGKey(666)
    discretise_ou_sde, cond_score_t_0, simulate_cond_forward = make_ou_sde(a, b)

    # Test discretise_ou_sde
    t = 100.
    F, Q = discretise_ou_sde(t)
    npt.assert_almost_equal(F, 0.)
    npt.assert_almost_equal(Q, stationary_var)

    # Test cond_score_t_0
    x = jnp.array(2.2)
    x0 = jnp.array(1.1)
    computed_score = cond_score_t_0(x, 5., x0)
    true_score = jax.grad(jax.scipy.stats.norm.logpdf)(x,
                                                       x0 * discretise_ou_sde(5.)[0],
                                                       jnp.sqrt(discretise_ou_sde(5.)[1]))
    npt.assert_almost_equal(computed_score, true_score)

    # Test simulate_cond_forward
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=100000)
    x0 = jnp.array([1.1])
    ts = jnp.linspace(0., 20., 21)
    paths = jax.vmap(simulate_cond_forward, in_axes=[0, None, None])(keys, x0, ts)
    approx_stationary_mean = jnp.mean(paths[:, -1], axis=0)
    approx_stationary_var = jnp.var(paths[:, -1], axis=0)
    npt.assert_allclose(approx_stationary_mean, stationary_mean, atol=1e-2)
    npt.assert_allclose(approx_stationary_var, stationary_var, rtol=1e-2)
