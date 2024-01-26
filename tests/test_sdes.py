import jax
import jax.numpy as jnp
import numpy.testing as npt
import flax.linen as nn
import optax

from fbs.nn import sinusoidal_embedding
from fbs.nn.models import make_simple_st_nn
from fbs.sdes.linear import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE, StationaryExpLinearSDE, \
    make_linear_sde_law_loss, make_ou_sde, make_ou_score_matching_loss
from fbs.sdes.simulators import reverse_simulator

jax.config.update("jax_enable_x64", True)


def test_linear_sdes():
    a, b = -0.5, 1.
    stationary_mean, stationary_var = 0., b ** 2 / (2 * -a)
    T = 40.

    const_sde = StationaryConstLinearSDE(a=a, b=b)
    lin_sde = StationaryLinLinearSDE(beta_min=0., beta_max=20., t0=0., T=T)
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
        loss_fn = make_linear_sde_law_loss(sde, nn_score, 0., T, 10)
        loss = loss_fn(None, key, x0.reshape(1, 1))
        npt.assert_almost_equal(loss, 0.)


def test_linlin_sde():
    """We use linear SDEs a lot, test it more properly.
    """


def test_cross_check():
    # Check the implementation with a reference implementation.
    key = jax.random.PRNGKey(666)
    a, b = -1., 1.
    score_fn = lambda u, t, param: u * t
    ou_discretise_sde, ou_cond_score_t_0, ou_simulate_cond_forward = make_ou_sde(a, b)
    ou_loss = make_ou_score_matching_loss(a, b, score_fn, t0=0., T=2., nsteps=100, random_times=True)

    sde = StationaryConstLinearSDE(a=a, b=b)
    linear_discretise_sde, linear_cond_score_t_0, linear_simulate_cond_forward = make_linear_sde(sde)
    lin_loss = make_linear_sde_law_loss(sde, score_fn, t0=0., T=2., nsteps=100, random_times=True)

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
    """Test if optimising the score loss can give correct results for Gaussian models.
    """
    key = jax.random.PRNGKey(666)

    a, b = -0.5, 1.
    stationary_mean = 0.
    stationary_var = b ** 2 / (2 * -a)
    sde = StationaryConstLinearSDE(a=a, b=b)

    def true_score(x, _):
        logpdf = lambda x_: jnp.sum(jax.scipy.stats.norm.logpdf(x_, stationary_mean, jnp.sqrt(stationary_var)))
        return jax.grad(logpdf)(x)

    class MLP(nn.Module):

        @nn.compact
        def __call__(self, x, t):
            time_emb = sinusoidal_embedding(t / dt, out_dim=32)
            time_emb = nn.Dense(features=64, kernel_init=nn.initializers.xavier_normal())(time_emb)
            time_emb = nn.gelu(time_emb)
            time_emb = nn.Dense(features=32, kernel_init=nn.initializers.xavier_normal())(time_emb)

            x = nn.Dense(features=64, kernel_init=nn.initializers.xavier_normal())(x)
            x = nn.gelu(x)
            x = nn.Dense(features=16, kernel_init=nn.initializers.xavier_normal())(x)

            z = x * time_emb[:16] + time_emb[16:]
            z = nn.Dense(features=16, kernel_init=nn.initializers.xavier_normal())(z)
            z = nn.gelu(z)
            z = nn.Dense(features=16, kernel_init=nn.initializers.xavier_normal())(z)
            z = nn.gelu(z)
            z = nn.Dense(features=2, kernel_init=nn.initializers.xavier_normal())(z)

            return jnp.squeeze(z)

    nsamples = 100
    t0, T = 0., 1.
    nsteps = 100
    dt = T / nsteps

    key, subkey = jax.random.split(key)
    _, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                       dim_in=2, batch_size=nsamples,
                                                       nn_model=MLP())

    loss_fn = make_linear_sde_law_loss(sde, nn_score, t0=t0, T=T, nsteps=nsteps, random_times=True)

    @jax.jit
    def optax_kernel(param_, opt_state_, key_, xy0s_):
        loss_, grad = jax.value_and_grad(loss_fn)(param_, key_, xy0s_)
        updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
        param_ = optax.apply_updates(param_, updates)
        return param_, opt_state_, loss_

    schedule = optax.exponential_decay(1e-2, 10, .95)
    optimiser = optax.adam(learning_rate=schedule)
    param = array_param
    opt_state = optimiser.init(param)

    for i in range(1000):
        key, subkey = jax.random.split(key)
        samples = stationary_mean + jnp.sqrt(stationary_var) * jax.random.normal(subkey, (nsamples, 2))
        key, subkey = jax.random.split(key)
        param, opt_state, loss = optax_kernel(param, opt_state, subkey, samples)

    learnt_score_fn = lambda x, t: nn_score(x[None, :], t, param)
    xs = jnp.mgrid[-1:1:0.1, -1:1:0.1].reshape(-1, 2)
    ts = jnp.linspace(0., 1., 10)

    approx_score_vals = jax.vmap(jax.vmap(learnt_score_fn, in_axes=[0, None]), in_axes=[None, 0])(xs, ts)
    true_score_vals = jax.vmap(jax.vmap(true_score, in_axes=[0, None]), in_axes=[None, 0])(xs, ts)
    err = jnp.mean((approx_score_vals - true_score_vals) ** 2)
    npt.assert_allclose(err, 0., atol=1e-2)


def test_reverse_simulators():
    """Simulating the reversal of a stationary process should also be stationary.
    """
    a, b = -0.5, 1.

    def drift(x, _):
        return -0.5 * x

    def dispersion(_):
        return 1.

    def score(x, _):
        return jax.grad(jax.scipy.stats.norm.logpdf)(x, stationary_mean, jnp.sqrt(stationary_var))

    def rev_simu(key_, u0):
        return reverse_simulator(key_, u0, ts, score, drift, dispersion)

    stationary_mean = 0.
    stationary_var = b ** 2 / (2 * -a)

    ts = jnp.linspace(0, 1., 2 ** 10 + 1)
    key = jax.random.PRNGKey(666)
    keys = jax.random.split(key, num=100000)

    u0s = stationary_mean + jnp.sqrt(stationary_var) * jax.random.normal(key, (100000,))
    uTs = jax.vmap(rev_simu, in_axes=[0, 0])(keys, u0s)

    npt.assert_allclose(jnp.mean(uTs), stationary_mean, atol=2)
    npt.assert_allclose(jnp.var(uTs), stationary_var, rtol=2)
