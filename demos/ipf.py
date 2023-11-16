import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from flax import linen as nn
from fbs.nn.utils import make_nn_with_time
from fbs.dsb import ipf_loss, simulate_discrete_time

jax.config.update("jax_enable_x64", True)

# Config
nsamples = 2000
niters = 200
key = jax.random.PRNGKey(666)
nn_float = jnp.float64
nn_param_init = nn.initializers.xavier_normal()

dt = 0.01
nsteps = 50
T = nsteps * dt
ts = jnp.linspace(0, T, nsteps + 1)


# Toy data
def sample_x0s(_key):
    _, _subkey = jax.random.split(_key)
    ss = jax.random.normal(_subkey, (nsamples, 2)) @ jnp.linalg.cholesky(jnp.array([[0.5, 0.2],
                                                                                    [0.2, 0.5]]))
    return jnp.array([0., 0.]) + ss


def sample_xTs(_key):
    _key, _subkey = jax.random.split(_key)
    _c1 = jnp.array([0., 2.]) + 0.4 * jax.random.normal(_subkey, (int(nsamples / 2), 2))
    _key, _subkey = jax.random.split(_key)
    _c2 = jnp.array([0., -2.]) + 0.4 * jax.random.normal(_subkey, (int(nsamples / 2), 2))
    return jnp.concatenate([_c1, _c2], axis=0)


key, subkey = jax.random.split(key)
x0s = sample_x0s(subkey)
key, subkey = jax.random.split(key)
xTs = sample_xTs(subkey)


# Reference measure Brownian motion
sigma = 1.


# Neural network construction
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        return jnp.squeeze(x)


mlp = MLP()
key, subkey = jax.random.split(key)
init_param_fwd, _, nn_fwd = make_nn_with_time(mlp, dim_in=2, batch_size=10, key=subkey)
key, subkey = jax.random.split(key)
init_param_bwd, _, nn_bwd = make_nn_with_time(mlp, dim_in=2, batch_size=10, key=subkey)

schedule = optax.exponential_decay(1e-2, niters, 0.9)
optimiser = optax.adam(learning_rate=schedule)
f_param = init_param_fwd
b_param = init_param_bwd


# The first IPF iteration
def init_f(x, t, _): return x


# @jax.jit
def optax_kernel(_r_param, _opt_state, _f_fn, _f_param, _ts, _init_samples, _key):
    _, _subkey = jax.random.split(_key)
    _loss, grad = jax.value_and_grad(ipf_loss)(_r_param, nn_bwd, nn_fwd, _f_param, _init_samples, _ts, sigma, _subkey)
    updates, _opt_state = optimiser.update(grad, _opt_state, _r_param)
    _b_param = optax.apply_updates(_r_param, updates)
    return _b_param, _opt_state, _loss


opt_state = optimiser.init(b_param)

for i in range(niters):
    key, subkey = jax.random.split(key)
    init_samples = sample_x0s(subkey)
    key, subkey = jax.random.split(key)
    b_param, opt_state, loss = optax_kernel(b_param, opt_state, init_f, _, ts, init_samples, subkey)
    print(f'i: {i}, loss: {loss}')
