import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.sdes.linear import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE, StationaryExpLinearSDE

jax.config.update("jax_enable_x64", True)


def test_linear_sdes():
    a, b = -0.8, 2.1
    stationary_mean, stationary_var = 0., b ** 2 / (2 * -a)
    T = 20.

    const_sde = StationaryConstLinearSDE(a=a, b=b)
    lin_sde = StationaryLinLinearSDE(a=a, b=b)
    exp_sde = StationaryExpLinearSDE(a=a, b=b, c=1.5, z=2.)

    key = jax.random.PRNGKey(666)
    for sde in (const_sde, lin_sde, exp_sde):
        discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

        # Test discretisation
        F, Q = discretise_linear_sde(T, 0.)
        npt.assert_almost_equal(F, 0.)
        npt.assert_almost_equal(Q, stationary_var)

        # Test score computation
        x = jnp.array(2.2)
        x0 = jnp.array(1.1)
        computed_score = cond_score_t_0(x, 5., x0, 2.2)
        true_score = jax.grad(jax.scipy.stats.norm.logpdf)(x,
                                                           x0 * discretise_linear_sde(5., 2.2)[0],
                                                           jnp.sqrt(discretise_linear_sde(5., 2.2)[1]))
        npt.assert_almost_equal(computed_score, true_score)

        # Test simulation
        _, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 10000)
        x0s = stationary_mean + jnp.sqrt(stationary_var) * jax.random.normal(subkey, (10000, 1))
        ts = jnp.linspace(0., 10., 3)
        paths = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(keys, x0s, ts)
        computed_mean = jnp.mean(paths, axis=0)[:, 0]
        computed_variance = jnp.var(paths, axis=0)[:, 0]
        for m, v in zip(computed_mean, computed_variance):
            npt.assert_almost_equal(m, stationary_mean, decimal=2)
            npt.assert_allclose(v, stationary_var, rtol=3e-2)
