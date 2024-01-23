import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.sdes.linear import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE, StationaryExpLinearSDE, \
    make_linear_sde_score_matching_loss, make_ou_sde, make_ou_score_matching_loss

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

        # Test loss
        nn_score = lambda u, t, param: cond_score_t_0(u, t, x0, 0.)
        loss_fn = make_linear_sde_score_matching_loss(sde, nn_score, 0., T, 10)
        loss = loss_fn(None, key, x0.reshape(1, 1))
        npt.assert_almost_equal(loss, 0.)


def test_cross_check():
    # Check the implementation with a reference implementation.
    key = jax.random.PRNGKey(666)
    a, b = -1., 1.
    score_fn = lambda u, t, param: u * t
    ou_discretise_sde, ou_cond_score_t_0, ou_simulate_cond_forward = make_ou_sde(a, b)
    ou_loss = make_ou_score_matching_loss(a, b, score_fn, t0=0., T=2., nsteps=100, random_times=True)

    sde = StationaryConstLinearSDE(a=a, b=b)
    linear_discretise_sde, linear_cond_score_t_0, linear_simulate_cond_forward = make_linear_sde(sde)
    lin_loss = make_linear_sde_score_matching_loss(sde, score_fn, t0=0., T=2., nsteps=100, random_times=True)

    F_ou, Q_ou = ou_discretise_sde(1.)
    F_l, Q_l = linear_discretise_sde(1., 0.)
    npt.assert_equal(F_ou, F_l)
    npt.assert_equal(Q_ou, Q_l)

    s_ou = ou_cond_score_t_0(2.2, 1.1, 1.5)
    l_ou = linear_cond_score_t_0(2.2, 1.1, 1.5, 0.)
    npt.assert_equal(s_ou, l_ou)

    ts = jnp.linspace(0., 2., 20)
    path_ou = ou_simulate_cond_forward(key, jnp.array([2.5]), ts)
    path_lin = linear_simulate_cond_forward(key, jnp.array([2.5]), ts)
    npt.assert_array_equal(path_ou, path_lin)

    loss_ou = ou_loss(None, key, jnp.ones((3, 2)))
    loss_lin = lin_loss(None, key, jnp.ones((3, 2)))
    npt.assert_equal(loss_ou, loss_lin)


def test_score_matching_routine():
    # Test if optimising the score loss can give correct results for Gaussian models.
    # TODO
    pass


def test_simulators():
    pass
