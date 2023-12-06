import matplotlib.pyplot as plt
import pytest
import math
import scipy
import jax
import jax.numpy as jnp
import jaxopt
import optax
import numpy.testing as npt
import flax.linen as nn
from fbs.dsb import ipf_loss, simulate_discrete_time
from fbs.nn.utils import make_nn_with_time

jax.config.update("jax_enable_x64", True)


def test_simulators():
    def f(x, t):
        return jnp.exp(-theta * dt) * x

    key = jax.random.PRNGKey(666)

    dt = 0.001
    nsteps = 2000
    T = nsteps * dt
    ts = jnp.linspace(0, T, nsteps + 1)

    theta = 1.
    sigma = jnp.sqrt(1 / (2 * theta) * (1 - jnp.exp(-2 * theta * dt))) / jnp.sqrt(dt)

    m0, v0 = 2., 2.
    true_m = jnp.exp(-theta * T) * m0
    true_var = jnp.exp(-2 * theta * T) * (2 * v0 * theta + jnp.exp(2 * theta * T) - 1) / (2 * theta)

    x0s = m0 + jnp.sqrt(v0) * jax.random.normal(key, (1000, 1))
    key, _ = jax.random.split(key)
    xTs = simulate_discrete_time(f, x0s, ts, sigma, key)[:, -1]

    npt.assert_allclose(jnp.mean(xTs, axis=0), true_m, atol=1e-2)
    npt.assert_allclose(jnp.var(xTs, axis=0), true_var, atol=1e-2)


def test_ipf_loss():
    key = jax.random.PRNGKey(666)

    nsamples = 1000
    x0s = jax.random.normal(key, (nsamples, 1))
    key, _ = jax.random.split(key)
    xTs = jax.random.normal(key, (nsamples, 1))

    dt = 0.01
    nsteps = 500
    T = nsteps * dt
    ts = jnp.linspace(0, T, nsteps + 1)
    sigma = jnp.sqrt((1 - jnp.exp(-dt)) / dt)

    def f(x, t, _):
        return jnp.exp(-0.5 * dt) * x

    def b(x, t, _):
        return f(x, t, _)

    key, _ = jax.random.split(key)
    loss1 = ipf_loss(_, b, f, _, x0s, ts, sigma, key)
    loss2 = ipf_loss(_, f, b, _, xTs, ts, sigma, key)
    npt.assert_allclose(loss1, loss2, rtol=1e-3)

    def approx_b(x, t, param):
        return jnp.exp(-param * dt) * x

    def obj_func(param):
        return ipf_loss(param, approx_b, f, _, x0s, ts, sigma, key)

    init_param = jnp.array(2.)
    opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
    opt_params, opt_state = opt_solver.run(init_param)

    npt.assert_allclose(opt_params, 0.5, atol=1e-2)


def test_ipf_one_pass():
    """Test one pass of IPF to see if the reverse can be learnt correctly.
    """
    key = jax.random.PRNGKey(666)
    nsamples_train = 100
    nsamples_test = 1000

    # x0s
    def x0s_sampler(_key, n_samples):
        _key, _subkey = jax.random.split(_key)
        _g1 = -1.5 + 0.4 * jax.random.normal(_subkey, (n_samples * 2, 1))
        _key, _subkey = jax.random.split(_key)
        _g2 = 1.5 + 0.4 * jax.random.normal(_subkey, (n_samples, 1))
        return jnp.concatenate([_g1, _g2], axis=0)

    def f(x, t, *args, **kwargs):
        return x

    sigma = 1.
    dt = 0.1
    nsteps = 10
    T = dt * nsteps
    ts = jnp.linspace(0, T, nsteps + 1)
    T = dt * nsteps
    nn_float = jnp.float64
    nn_param_init = nn.initializers.xavier_normal()

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=16, param_dtype=nn_float, kernel_init=nn_param_init)(x)
            x = nn.relu(x)
            x = nn.Dense(features=8, param_dtype=nn_float, kernel_init=nn_param_init)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1, param_dtype=nn_float, kernel_init=nn_param_init)(x)
            return jnp.squeeze(x)

    mlp = MLP()
    key, subkey = jax.random.split(key)
    init_param_fwd, _, nn_fwd = make_nn_with_time(mlp, dim_in=1, batch_size=nsamples_train, time_scale=1, key=subkey)
    key, subkey = jax.random.split(key)
    init_param_bwd, _, nn_bwd = make_nn_with_time(mlp, dim_in=1, batch_size=nsamples_train, time_scale=1, key=subkey)

    niters = 200
    schedule = optax.cosine_decay_schedule(1e-2, 1, .95)
    optimiser = optax.adam(learning_rate=schedule)
    f_param = init_param_fwd
    b_param = init_param_bwd

    @jax.jit
    def optax_kernel(_b_param, _opt_state, _key):
        _key, _subkey = jax.random.split(_key)
        _x0s = x0s_sampler(_subkey, nsamples_train)
        _, _subkey = jax.random.split(_key)
        _loss, grad = jax.value_and_grad(ipf_loss)(_b_param, nn_bwd, f, _, _x0s, ts, sigma, _subkey)
        updates, _opt_state = optimiser.update(grad, _opt_state, _b_param)
        _b_param = optax.apply_updates(_b_param, updates)
        return _b_param, _opt_state, _loss

    key, subkey = jax.random.split(key)
    x0s = x0s_sampler(subkey, nsamples_test)

    key, subkey = jax.random.split(key)
    xTs = simulate_discrete_time(f, x0s, ts, sigma, subkey)[:, -1]

    # Learning
    opt_state = optimiser.init(b_param)

    for i in range(niters):
        key, subkey = jax.random.split(key)
        b_param, opt_state, loss = optax_kernel(b_param, opt_state, subkey)

    key, subkey = jax.random.split(key)
    approx_x0s = simulate_discrete_time(nn_bwd, xTs, T - ts, sigma, subkey, b_param)[:, -1, :]

    npt.assert_allclose(scipy.stats.wasserstein_distance(x0s[:, 0], approx_x0s[:, 0]), 0, atol=1e-1)