"""
Try conditional sampling
"""
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
from fbs.utils import discretise_lti_sde
from fbs.filters import bootstrap_filter, stratified
from functools import partial

# General configs
nsamples = 1000
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

dt = 0.001
nsteps = 2000
T = nsteps * dt
ts = jnp.linspace(0, T, nsteps + 1)

# Target distribution \pi(x, y)
m0 = jnp.array([1., -1.])
cov0 = jnp.array([[2., 0.5],
                  [0.5, 1.2]])

y0 = 0.
true_cond_m = m0[0] + cov0[0, 1] / cov0[1, 1] * (y0 - m0[1])
true_cond_var = cov0[0, 0] - cov0[0, 1] ** 2 / cov0[1, 1]

A = -0.5 * jnp.eye(2)
B = jnp.eye(2)


def forward_m_cov(t):
    F, Q = discretise_lti_sde(A, B, t)
    return F @ m0, F @ cov0 @ F.T + Q


def score(z, t):
    mt, covt = forward_m_cov(t)
    return jax.grad(jax.scipy.stats.multivariate_normal.logpdf, argnums=0)(z, mt, covt)


def simulate_forward(xy0, key_):
    def scan_body(carry, elem):
        xy = carry
        dw = elem

        xy = F_ @ xy + chol @ dw
        return xy, xy

    F_, Q_ = discretise_lti_sde(A, B, dt)
    chol = jnp.linalg.cholesky(Q_)
    dws = jnp.sqrt(dt) * jax.random.normal(key_, (nsteps, 2))
    return jnp.concatenate([xy0[None, :], jax.lax.scan(scan_body, xy0, dws)[1]], axis=0)


# Reference distribution \pi_ref
m_ref, cov_ref = forward_m_cov(T)


# The reverse process
def reverse_drift(uv, t):
    return -A @ uv + B @ score(uv, T - t)


def reverse_drift_u(u, v, t):
    uv = jnp.asarray([u, v])
    return (-A @ uv + B @ score(uv, T - t))[0]


def reverse_drift_v(v, u, t):
    uv = jnp.asarray([u, v])
    return (-A @ uv + B @ score(uv, T - t))[1]


# Test the implementation correctness. Only for testing purpose. Ignore this block!
key, subkey = jax.random.split(key)
uv0s = m_ref + jax.random.normal(subkey, (nsamples, 2)) @ jnp.linalg.cholesky(cov_ref)


def backward_euler(uv0, key_):
    def scan_body(carry, elem):
        uv = carry
        dw, t = elem

        uv += reverse_drift(uv, t) * dt + B @ dw
        return uv, None

    _, subkey_ = jax.random.split(key_)
    dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, 2))
    return jax.lax.scan(scan_body, uv0, (dws, ts[:-1]))[0]


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
approx_init_samples = jax.vmap(backward_euler, in_axes=[0, 0])(uv0s, keys)

print(jnp.mean(approx_init_samples, axis=0), m0)
print(jnp.cov(approx_init_samples, rowvar=False), cov0)


# Now particle filtering for the conditional sampling
def transition_sampler(us, v, t, key_):
    return (us + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us, v, t) * dt
            + math.sqrt(dt) * jax.random.normal(key_, us.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def measurement_cond_logpdf(v, u_prev, v_prev, t_prev):
    cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
    return jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt))


def init_sampler(key_, v0, nsamples_):
    cond_m = m_ref[0] + cov_ref[0, 1] / cov_ref[1, 1] * (v0 - m_ref[1])
    cond_var = cov_ref[0, 0] - cov_ref[0, 1] ** 2 / cov_ref[1, 1]
    return cond_m + jnp.sqrt(cond_var) * jax.random.normal(key_, (nsamples_,))


# Simulate forward pass for ys
key, subkey = jax.random.split(key)
xy0 = jnp.array([0., y0])
xys = simulate_forward(xy0, subkey)
vs = xys[::-1, 1]

key, subkey = jax.random.split(key)
pf_samples, _ = bootstrap_filter(transition_sampler, measurement_cond_logpdf,
                                 vs, ts, init_sampler, key, nsamples, resampling=stratified)
approx_cond_x0s = pf_samples[-1]
print(jnp.mean(approx_cond_x0s), true_cond_m)
print(jnp.var(approx_cond_x0s), true_cond_var)
