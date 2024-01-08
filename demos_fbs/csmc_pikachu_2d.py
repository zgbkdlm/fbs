r"""
Conditional sampling on a 2D non-Gaussian target distribution.

X \in R^2
Y \in R
"""
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
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
y0 = 1.

T = 5
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)


# Target distribution with known conditional sampling scheme
def sampler_x_cond_y(key_, y):
    u = jnp.sqrt(jnp.array([[2., 0.],
                            [0., 1.]])) @ jax.random.normal(key_, (2,))
    return jnp.array([u[1] / (2. + y) + 0.1 * (u[0] + y) * u[0],
                      u[0] / (2. + y) + 0.1 * (u[1] + y) * u[1]])


def sampler_y(key_):
    return 0.5 * jax.random.normal(key_)


def sampler_xy(key_):
    y = sampler_y(key_)
    x = sampler_x_cond_y(jax.random.split(key_)[1], y)
    return jnp.hstack([x, y])


keys = jax.random.split(key, nsamples)
xs = jax.vmap(sampler_xy, in_axes=[0])(keys)

plt.scatter(xs[:, 0], xs[:, 1], s=1, alpha=0.5, label='p(x)')
plt.legend()
plt.show()

xs = jax.vmap(sampler_x_cond_y, in_axes=[0, None])(keys, y0)
plt.scatter(xs[:, 0], xs[:, 1], s=1, alpha=0.5, label=f'p(x | y = {y0})')
plt.legend()
plt.show()

# Define the forward noising process
A = -0.5 * jnp.eye(3)
B = jnp.eye(3)
gamma = B @ B.T


def cond_score_t_0(xy, t, xy0):
    F, Q = discretise_lti_sde(A, gamma, t)
    return jax.grad(jax.scipy.stats.multivariate_normal.logpdf)(xy, F @ xy0, Q)


def simulate_cond_forward(key_, xy0, ts_):
    def scan_body(carry, elem):
        xy = carry
        dw = elem

        xy = F @ xy + chol @ dw
        return xy, xy

    # F, Q = discretise_lti_sde(A, gamma, dt)
    # chol = jnp.linalg.cholesky(Q)
    # dws = jnp.sqrt(dt) * jax.random.normal(key_, (nsteps, 3))
    dts = jnp.diff(ts_)
    return jnp.concatenate([xy0[None, :], jax.lax.scan(scan_body, xy0, dws)[1]], axis=0)


def simulate_forward(key_):
    xy0 = sampler_xy(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], xy0)


# Score matching
batch_nsamples = 100
batch_nsteps = 100


def loss_fn(param_, key_):
    fwd_paths = jax.vmap(simulate_forward)()
