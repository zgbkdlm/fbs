import pytest
import math
import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.dsb import ipf_fwd_loss, ipf_bwd_loss, simulate_discrete_time

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


def test_ipf_fwd_bwd():
    key = jax.random.PRNGKey(666)

    nsamples = 1000
    x0s = jax.random.normal(key, (nsamples, 1))
    key, _ = jax.random.split(key)
    xTs = jax.random.normal(key, (nsamples, 1))

    dt = 0.01
    nsteps = 100
    T = nsteps * dt
    ts = jnp.linspace(0, T, nsteps + 1)
    sigma = jnp.sqrt((1 - jnp.exp(-dt)) / dt)

    def f(x, t, _):
        return jnp.exp(-0.5 * dt) * x

    def b(x, t, _):
        return f(x, t, _)

    key, _ = jax.random.split(key)
    loss_fwd = ipf_fwd_loss(_, _, f, b, x0s, ts, sigma, key)
    loss_bwd = ipf_bwd_loss(_, _, f, b, xTs, ts, sigma, key)
    print(loss_fwd, loss_bwd)
