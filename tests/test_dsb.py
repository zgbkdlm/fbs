import pytest
import math
import jax
import jax.numpy as jnp
import jaxopt
import numpy.testing as npt
from fbs.dsb import ipf_loss, simulate_discrete_time

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
    # TODO
