import jax
import jax.numpy as jnp
import numpy.testing as npt
import flax.linen as nn
import optax
import math
from fbs.nn import sinusoidal_embedding
from fbs.nn.models import make_simple_st_nn
from fbs.sdes.linear import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE, StationaryExpLinearSDE, \
    make_linear_sde_law_loss, make_ou_sde, make_ou_score_matching_loss
from fbs.sdes.simulators import reverse_simulator, doob_bridge_simulator, euler_maruyama
from fbs.dsb.base import ipf_loss_cont
from fbs.sdes.linear import make_gaussian_bw_sb
from functools import partial

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
    a, b = -0.5, 1.
    T = 40.
    ts = jnp.linspace(1e-5, T, 100)

    const_sde = StationaryConstLinearSDE(a=a, b=b)
    lin_sde = StationaryLinLinearSDE(beta_min=1., beta_max=1., t0=0., T=T)

    disc_const, _, _ = make_linear_sde(const_sde)
    disc_lin, _, _ = make_linear_sde(lin_sde)

    # They should be identical
    for i in range(2):
        npt.assert_allclose(disc_lin(ts, 0.)[i], disc_const(ts, 0.)[i])

    # Check if the result matches a closed-form result
    ts = jnp.linspace(1e-5, 1., 100)
    beta_min, beta_max = 1e-3, 3.
    lin_sde = StationaryLinLinearSDE(beta_min=beta_min, beta_max=beta_max, t0=0., T=1.)
    disc_lin, _, _ = make_linear_sde(lin_sde)
    alp = ts * beta_min + 0.5 * ts ** 2 * (beta_max - beta_min)
    true_m = jnp.exp(-0.5 * alp)
    true_var = 1 - jnp.exp(-alp)
    npt.assert_allclose(disc_lin(ts, 0.)[0], true_m)
    npt.assert_allclose(disc_lin(ts, 0.)[1], true_var)


def test_linlin_sde_statistics():
    T = 2
    nsteps = 200
    ts = jnp.linspace(0, T, nsteps + 1)

    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=5, t0=0., T=T)
    _, _, simulate_cond_forward = make_linear_sde(sde)

    x0 = jnp.array(2.)

    key = jax.random.PRNGKey(666)
    keys = jax.random.split(key, num=10000)

    def fwd_sampler(key_):
        return simulate_cond_forward(key_, x0, ts)

    paths = jax.vmap(fwd_sampler)(keys)

    npt.assert_allclose(jnp.mean(paths, axis=0), sde.mean(ts, ts[0], x0), rtol=6e-2)
    npt.assert_allclose(jnp.var(paths, axis=0), sde.variance(ts, ts[0]), rtol=6e-2)


def test_linlin_sde_bridge():
    T = 1
    nsteps = 100
    ts = jnp.linspace(0, T, nsteps + 1)

    sde = StationaryLinLinearSDE(beta_min=0.1, beta_max=2., t0=0., T=T)
    target = jnp.array(5.)
    x0 = jnp.array(1.)

    def simulator(key_):
        return doob_bridge_simulator(key_, sde, x0, target, ts,
                                     integration_nsteps=100,
                                     replace=False)[-1]

    key = jax.random.PRNGKey(666)
    keys = jax.random.split(key, num=20)
    terminal_vals = jax.vmap(simulator)(keys)
    npt.assert_allclose(terminal_vals, jnp.ones(20) * target, rtol=2e-2)


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


def test_gaussian_sb():
    """Test the Gaussian Schrodinger bridge.
    """
    key = jax.random.PRNGKey(666)

    mu0 = jnp.array([1.5, -1.8])
    cov0 = jnp.array([[1., 0.3],
                      [0.3, 1.5]])
    mu1 = jnp.array([-1., 2.2])
    cov1 = jnp.array([[0.5, -0.2],
                      [-0.2, 0.7]])

    marginal_mean, marginal_cov, drift = make_gaussian_bw_sb(mu0, cov0, mu1, cov1, sig=1.)

    # Test marginals
    npt.assert_allclose(mu0, marginal_mean(0.), rtol=1e-8)
    npt.assert_allclose(mu1, marginal_mean(1.), rtol=1e-8)

    npt.assert_allclose(cov0, marginal_cov(0.), rtol=1e-8)
    npt.assert_allclose(cov1, marginal_cov(1.), rtol=1e-8)

    # Test drift
    t0 = 0.
    T = 1.
    nsteps = 100
    ts = jnp.linspace(t0, T, nsteps + 1)
    nsamples = 10000

    def dispersion(t):
        return 1.

    def terminal_simulator(key_, x0):
        return euler_maruyama(key_, x0, ts, drift, dispersion, integration_nsteps=10, return_path=False)

    key, subkey = jax.random.split(key)
    init_samples = mu0 + jax.random.normal(subkey, (nsamples, 2)) @ jnp.linalg.cholesky(cov0)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=nsamples)
    terminal_samples = jax.vmap(terminal_simulator, in_axes=[0, 0])(keys, init_samples)
    approx_m1 = jnp.mean(terminal_samples, axis=0)
    approx_cov1 = jnp.cov(terminal_samples, rowvar=False)

    npt.assert_allclose(mu1, approx_m1, rtol=1e-1)
    npt.assert_allclose(cov1, approx_cov1, rtol=1e-1)

    # Test if the reverse of the reverse is the forward
    def score(x, t):
        mt, covt = marginal_mean(t), marginal_cov(t)
        chol = jax.scipy.linalg.cho_factor(covt)
        return -jax.scipy.linalg.cho_solve(chol, x - mt)

    def reverse_drift(x, t):
        return -drift(x, 1 - t) + score(x, 1 - t)

    def reverse_reverse_drift(x, t):
        return -reverse_drift(x, t) + score(x, t)

    npt.assert_allclose(reverse_reverse_drift(mu0, 0.5), drift(mu0, 0.5))


def test_sb_loss():
    """Test the Schrodinger bridge loss function.
    """
    key = jax.random.PRNGKey(666)

    mu0 = jnp.array([1.5, -1.8])
    cov0 = jnp.array([[1., 0.3],
                      [0.3, 1.5]])
    mu1 = jnp.array([-1., 2.2])
    cov1 = jnp.array([[0.5, -0.2],
                      [-0.2, 0.7]])

    t0 = 0.
    T = 1.
    ts = jnp.linspace(t0, T, 100)
    nsamples = 10000

    marginal_mean, marginal_cov, sb_drift = make_gaussian_bw_sb(mu0, cov0, mu1, cov1, sig=1.)

    # Test the loss for learning the backward
    key, subkey = jax.random.split(key)
    init_samples = mu0 + jax.random.normal(subkey, (nsamples, 2)) @ jnp.linalg.cholesky(cov0)

    def fwd_drift(x, t, p):
        return sb_drift(x, t) * p

    def score(x, t):
        mt, covt = marginal_mean(t), marginal_cov(t)
        chol = jax.scipy.linalg.cho_factor(covt)
        return -jax.scipy.linalg.cho_solve(chol, x - mt)

    def reverse_drift(x, t, p):
        return -fwd_drift(x, T - t, 1.) + score(x, T - t) + p

    key, subkey = jax.random.split(key)

    def loss_fn_bwd(p):
        return ipf_loss_cont(subkey, p, 1., init_samples, ts,
                             jax.vmap(reverse_drift, in_axes=[0, None, None]),
                             jax.vmap(fwd_drift, in_axes=[0, None, None]))

    # The loss function should be at a stationary point when p = 0 of the backward drift
    npt.assert_allclose(jax.grad(loss_fn_bwd)(jnp.array(0.)), 0., atol=1e-3)

    # Test the loss for learning the forward
    key, subkey = jax.random.split(key)
    terminal_samples = mu1 + jax.random.normal(subkey, (nsamples, 2)) @ jnp.linalg.cholesky(cov1)

    def loss_fn_fwd(p):
        return ipf_loss_cont(subkey, p, 0., terminal_samples, ts,
                             jax.vmap(fwd_drift, in_axes=[0, None, None]),
                             jax.vmap(reverse_drift, in_axes=[0, None, None]))

    print(jax.grad(loss_fn_fwd)(jnp.array(1.)))
    npt.assert_allclose(jax.grad(loss_fn_fwd)(jnp.array(0.)), 0., atol=1e-2)
