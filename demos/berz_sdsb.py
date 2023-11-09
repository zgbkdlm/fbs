"""
Test supervised DSB.
"""
import argparse
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from fbs.nn.utils import make_nn_with_time
from functools import partial

parser = argparse.ArgumentParser(description='Test supervised DSB on Berzelius.')
parser.add_argument('--train', action='store_true', help='Whether to train or test.')
args = parser.parse_args()

# General configs
nsamples = 10000
jax.config.update("jax_enable_x64", True)
nn_param_init = nn.initializers.xavier_normal()
key = jax.random.PRNGKey(666)

dt = 0.01
nsteps = 200
T = nsteps * dt
ts = jnp.linspace(dt, T, nsteps)


# Neural network construction
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=100, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=20, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=2, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
        return jnp.squeeze(x)


mlp = MLP()
key, subkey = jax.random.split(key)
init_param, _, nn_eval = make_nn_with_time(mlp, dim_in=2, batch_size=10, key=subkey)

# Draw samples on the two sides
np.random.seed(999)


@partial(jax.vmap, in_axes=[0])
def f1(z):
    """x = f1(z)"""
    u = jnp.sin(0.5 * z)
    return jnp.array([(u[0] + u[1]) / 2, u[0] * u[1]])


@partial(jax.vmap, in_axes=[0])
def f2(x):
    """y = f2(x)"""
    u = jnp.tanh(2 * x)
    return jnp.array([u[0] * u[1], (u[0] + u[1]) / 2])


key, subkey = jax.random.split(key)
zs = jax.random.normal(subkey, (nsamples, 2))
xs = f1(zs)
ys = f2(xs)


# Define the noising forward reference
def drift(xt, xT, t):
    """Drift of Brownian bridge SDE.
    """
    return (xT - xt) / (T - t)


def simulate_bridge_forward(x0, xT, _key):
    def scan_body(carry, elem):
        x = carry
        t, dw = elem
        x = x + drift(x, xT, t) * dt + dw
        return x, x

    _, _subkey = jax.random.split(_key)
    rnds = math.sqrt(dt) * jax.random.normal(_subkey, (nsteps - 1, 2))
    traj = jax.lax.scan(scan_body, x0, (ts[:-1], rnds))[1]
    return jnp.concatenate([traj, xT.reshape(1, -1)])


scales = (T - ts) * ts / (T * dt) + 1e-8


def cond_pdf_t_0(xt, t, xT, x0):
    mt, vt = x0 + t / T * (xT - x0), (T - t) * t / T + 1e-8
    return jnp.sum(jax.scipy.stats.norm.logpdf(xt, mt, jnp.sqrt(vt)), axis=-1)


@partial(jax.vmap, in_axes=[0, None, 0, 0])
@partial(jax.vmap, in_axes=[0, 0, None, None])
def cond_score_t_0(xt, t, xT, x0):
    return jax.grad(cond_pdf_t_0)(xt, t, xT, x0)


if args.train:
    def loss_fn(_param, _key, overfit=False):
        if overfit:
            _xs = xs
            _ys = ys
            _keys = jax.random.split(_key, num=nsamples)
        else:
            _key, _subkey = jax.random.split(_key)
            _zs = jax.random.normal(_subkey, (nsamples, 2))
            _xs = f1(zs)
            _ys = f2(xs)
            _keys = jax.random.split(_subkey, num=nsamples)
        forward_paths = jax.vmap(simulate_bridge_forward, in_axes=[0, 0, 0])(_xs, _ys, _keys)  # (nsamples, nsteps, 2)
        errs = (jax.vmap(jax.vmap(nn_eval,
                                  in_axes=[0, 0, None]),
                         in_axes=[0, None, None])(forward_paths, ts, _param) -
                cond_score_t_0(forward_paths, ts, _ys, _xs)) ** 2  # (nsamples, nsteps, 2)
        return jnp.sum(jnp.mean(errs, 0))


    @jax.jit
    def opt_step_kernel(_param, _opt_state, _key):
        _loss, grad = jax.value_and_grad(loss_fn)(_param, _key)
        updates, _opt_state = optimiser.update(grad, _opt_state, _param)
        _param = optax.apply_updates(_param, updates)
        return _param, _opt_state, _loss


    optimiser = optax.adam(learning_rate=1e-2)
    opt_state = optimiser.init(init_param)
    param = init_param

    for i in range(10000):
        key, subkey = jax.random.split(key)
        param, opt_state, loss = opt_step_kernel(param, opt_state, subkey)
        print(f'i: {i}, loss: {loss}')

    np.save('./param_sdsb', param)
else:
    param = np.load('./param_sdsb.npy')


# Backward sampling
def simulate_backward(xT, _key):
    def scan_body(carry, elem):
        x = carry
        t, dw = elem

        x = x + (-drift(x, xT, T - t) + nn_eval(x, T - t, param)) * dt + dw
        return x, x

    _, _subkey = jax.random.split(_key)
    dws = jnp.sqrt(dt) * jax.random.normal(_subkey, (nsteps, 2))
    return jax.lax.scan(scan_body, xT, (ts, dws))[1]


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
backward_traj = jax.vmap(simulate_backward, in_axes=[0, 0])(ys, keys)
