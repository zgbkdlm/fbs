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
import flax.linen as nn
from fbs.nn import sinusoidal_embedding
from fbs.nn.models import make_simple_st_nn
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
        dt_, rnd = elem

        F, Q = discretise_lti_sde(A, gamma, dt_)
        xy = F @ xy + jnp.linalg.cholesky(Q) @ rnd
        return xy, xy

    dts = jnp.diff(ts_)
    rnds = jax.random.normal(key_, (dts.shape[0], 3))
    return jnp.concatenate([xy0[None, :], jax.lax.scan(scan_body, xy0, (dts, rnds))[1]], axis=0)


def simulate_forward(key_, ts_):
    xy0 = sampler_xy(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], xy0, ts_)


# Score matching
batch_nsamples = 100
batch_nsteps = 100
ntrains = 1000
nn_param_init = nn.initializers.xavier_normal()


class MLP(nn.Module):
    @nn.compact
    def __call__(self, xy, t):
        # Spatial part
        xy = nn.Dense(features=16, param_dtype=jnp.float64, kernel_init=nn_param_init)(xy)
        xy = nn.relu(xy)
        xy = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(xy)

        # Temporal part
        t = sinusoidal_embedding(t, out_dim=128)
        t = nn.Dense(features=16, param_dtype=jnp.float64, kernel_init=nn_param_init)(t)
        t = nn.relu(t)
        t = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(t)

        z = jnp.concatenate([xy, t], axis=-1)
        z = nn.Dense(features=32, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=3, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
        return jnp.squeeze(z)


key, subkey = jax.random.split(key)
_, _, array_param, _, nn_score = make_simple_st_nn(key,
                                                   dim_x=3, batch_size=batch_nsamples,
                                                   mlp=MLP())


def loss_fn(param_, key_):
    key_ts, key_fwd, key__ = jax.random.split(key_, num=3)
    batch_ts = jnp.hstack([0.,
                           jnp.sort(jax.random.uniform(key_ts, (batch_nsteps - 2,), minval=0., maxval=T)),
                           T])
    fwd_paths = jax.vmap(simulate_forward, in_axes=[0, None])(jax.random.split(key_fwd, num=batch_nsamples),
                                                              batch_ts)
    nn_evals = jax.vmap(jax.vmap(nn_score,
                                 in_axes=[0, 0, None]),
                        in_axes=[0, None, None])(fwd_paths[:, 1:], batch_ts[1:], param_)
    cond_score_evals = jax.vmap(jax.vmap(cond_score_t_0,
                                         in_axes=[0, 0, None]),
                                in_axes=[0, None, 0])(fwd_paths[:, 1:], batch_ts[1:], fwd_paths[:, 0])
    return jnp.sum(jnp.mean((nn_evals - cond_score_evals) ** 2, axis=0))


@jax.jit
def optax_kernel(param_, opt_state_, key_):
    loss_, grad = jax.value_and_grad(loss_fn)(param_, key_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state_, loss_


optimiser = optax.adam(learning_rate=optax.cosine_decay_schedule(1e-2, 10, .95))
param = array_param
opt_state = optimiser.init(param)

for i in range(ntrains):
    key, subkey = jax.random.split(key)
    param, opt_state, loss = optax_kernel(param, opt_state, subkey)
    print(f'i: {i}, loss: {loss}')


# Verify if the score function is learnt properly
def reverse_drift(uv, t):
    return -A @ uv + gamma @ nn_score(uv, T - t, param)


def reverse_drift_u(u, v, t):
    uv = jnp.concatenate([u, v], axis=-1)
    return (-A @ uv + gamma @ nn_score(uv, T - t, param))[:2]


def reverse_drift_v(v, u, t):
    uv = jnp.concatenate([u, v], axis=-1)
    return (-A @ uv + gamma @ nn_score(uv, T - t, param))[-1]


def backward_euler(uv0, key_):
    def scan_body(carry, elem):
        uv = carry
        dw, t = elem

        uv += reverse_drift(uv, t) * dt + B @ dw
        return uv, None

    _, subkey_ = jax.random.split(key_)
    dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, 3))
    return jax.lax.scan(scan_body, uv0, (dws, ts[:-1]))[0]
