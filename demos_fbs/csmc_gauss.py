"""
Try conditional sampling using the cSMC approach.
"""
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
from fbs.utils import discretise_lti_sde
from fbs.filters.csmc.csmc import csmc_kernel
from fbs.filters.csmc.resamplings import killing
from functools import partial

# General configs
nparticles = 100
nsamples = 1000
burn_in = 100
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

T = 5
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# Target distribution \pi(x, y)
m0 = jnp.array([1., -1.])
cov0 = jnp.array([[2., 0.5],
                  [0.5, 1.2]])

y0 = 5.
true_cond_m = m0[0] + cov0[0, 1] / cov0[1, 1] * (y0 - m0[1])
true_cond_var = cov0[0, 0] - cov0[0, 1] ** 2 / cov0[1, 1]

A = -0.5 * jnp.eye(2)
B = jnp.array([[1., 0.],
               [0., 1.]])
gamma = B @ B.T
chol_gamma = jnp.linalg.cholesky(gamma)


def forward_m_cov(t):
    F, Q = discretise_lti_sde(A, gamma, t)
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

    F_, Q_ = discretise_lti_sde(A, gamma, dt)
    chol = jnp.linalg.cholesky(Q_)
    dws = jax.random.normal(key_, (nsteps, 2))
    return jnp.concatenate([xy0[None, :], jax.lax.scan(scan_body, xy0, dws)[1]], axis=0)


# Reference distribution \pi_ref
m_ref, cov_ref = forward_m_cov(T)


# The reverse process
def reverse_drift(uv, t):
    return -A @ uv + gamma @ score(uv, T - t)


def reverse_drift_u(u, v, t):
    uv = jnp.asarray([u, v])
    return (-A @ uv + gamma @ score(uv, T - t))[0]


def reverse_drift_v(v, u, t):
    uv = jnp.asarray([u, v])
    return (-A @ uv + gamma @ score(uv, T - t))[1]


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


def transition_sampler(us, v, t, key_):
    return (us + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us, v, t) * dt
            + math.sqrt(dt) * chol_gamma[0, 0] * jax.random.normal(key_, us.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v, t):
    return jax.scipy.stats.norm.logpdf(u,
                                       u_prev + reverse_drift_u(u_prev, v, t) * dt,
                                       math.sqrt(dt) * chol_gamma[0, 0])


@partial(jax.vmap, in_axes=[None, 0, None, None])
def likelihood_logpdf(v, u_prev, v_prev, t_prev):
    cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
    return jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * B[1, 1])


def fwd_sampler(key_, x0):
    xy0 = jnp.array([x0, y0])
    return simulate_forward(xy0, key_)


@jax.jit
def gibbs_kernel(key_, xs_, us_star_, bs_star_):
    key_fwd, key_csmc = jax.random.split(key_)
    path_xy = fwd_sampler(key_fwd, xs_[0])
    us, vs = path_xy[::-1, 0], path_xy[::-1, 1]

    def init_sampler(*_):
        return us[0] * jnp.ones(nparticles)

    def init_likelihood_logpdf(*_):
        return -math.log(nparticles) * jnp.ones(nparticles)

    us_star_next, bs_star_next = csmc_kernel(key_csmc,
                                             us_star_, bs_star_,
                                             vs, ts,
                                             init_sampler, init_likelihood_logpdf,
                                             transition_sampler, transition_logpdf,
                                             likelihood_logpdf,
                                             killing, nparticles,
                                             backward=True)
    xs_next = us_star_next[::-1]
    return xs_next, us_star_next, bs_star_next, bs_star_next != bs_star_


# Gibbs loop
key, subkey = jax.random.split(key)
xs = fwd_sampler(subkey, true_cond_m)[:, 0]
us_star = xs[::-1]
bs_star = jnp.zeros((nsteps + 1), dtype=int)

uss = np.zeros((nsamples, nsteps + 1))
xss = np.zeros((nsamples, nsteps + 1))
for i in range(nsamples):
    print(i)
    key, subkey = jax.random.split(key)
    xs, us_star, bs_star, acc = gibbs_kernel(subkey, xs, us_star, bs_star)
    xss[i], uss[i] = xs, us_star

plt.plot(uss[:, -1])
plt.show()

uss = uss[burn_in:]

key, subkey = jax.random.split(key)
plt.hist(uss[:, -1], density=True, bins=50, alpha=0.5, label=f'Approx. target posterior p(x | y={y0})')
plt.hist(true_cond_m + jnp.sqrt(true_cond_var) * jax.random.normal(subkey, (nsamples,)),
         density=True, bins=50, alpha=0.5, label=f'True target posterior p(x | y={y0})')
plt.legend()
plt.show()
