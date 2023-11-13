"""
Test supervised DSB.
"""
import argparse
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from flax import linen as nn
from fbs.nn.utils import make_nn_with_time

parser = argparse.ArgumentParser(description='Test supervised DSB on Berzelius.')
parser.add_argument('--train', action='store_true', help='Whether to train or test.')
args = parser.parse_args()

# General configs
nsamples = 800
jax.config.update("jax_enable_x64", True)
nn_float = jnp.float64
nn_param_init = nn.initializers.xavier_normal()
key = jax.random.PRNGKey(666)

dt = 0.01
nsteps = 400
T = nsteps * dt
ts = jnp.linspace(dt, T, nsteps)
ts_training = ts[:-1]


# Neural network construction
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=50, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=20, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2, param_dtype=nn_float, kernel_init=nn_param_init)(x)
        return jnp.squeeze(x)


mlp = MLP()
key, subkey = jax.random.split(key)
init_param, _, nn_eval = make_nn_with_time(mlp, dim_in=2, batch_size=10, key=subkey)

# Draw training samples
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=2)
g1 = jnp.array([3, 3]) + 0.2 * jax.random.normal(keys[0], (int(nsamples / 2), 2))
# g2 = jnp.array([-3, 3]) + 0.2 * jax.random.normal(keys[1], (int(nsamples / 4), 2))
g3 = jnp.array([-3, -3]) + 0.2 * jax.random.normal(keys[1], (int(nsamples / 2), 2))
# g4 = jnp.array([3, -3]) + 0.2 * jax.random.normal(keys[3], (int(nsamples / 4), 2))
xs = jnp.concatenate([g1, g3], axis=0)
ys = -5 / xs


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


scales = (T - ts) * ts / (T * dt) + 0


def cond_pdf_t_0(xt, t, xT, x0):
    mt, vt = x0 + t / T * (xT - x0), (T - t) * t / T
    return jnp.sum(jax.scipy.stats.norm.logpdf(xt, mt, jnp.sqrt(vt)), axis=-1)


def cond_score_t_0(xt, t, xT, x0):
    return jax.grad(cond_pdf_t_0)(xt, t, xT, x0)


def compute_errs(x0, xT, _param, _key):
    def scan_body(carry, elem):
        x, err = carry
        t, dw = elem

        x = x + drift(x, xT, t) * dt + dw
        err = err + jnp.sum((nn_eval(x, t, _param) - cond_score_t_0(x, t, xT, x0)) ** 2)
        return (x, err), None

    _, _subkey = jax.random.split(_key)
    rnds = math.sqrt(dt) * jax.random.normal(_subkey, (nsteps, 2))
    return jax.lax.scan(scan_body, (x0, 0.), (ts[:-1], rnds[:-1]))[0][1]


def loss_fn(_param, _key, overfit=False):
    if overfit:
        _xs = xs
        _ys = ys
    else:
        _key, _subkey = jax.random.split(_key)
        _keys = jax.random.split(_subkey, num=1)
        _g1 = jnp.array([3, 3]) + 0.2 * jax.random.normal(_keys[0], (int(nsamples / 2), 2))
        # _g2 = jnp.array([-3, 3]) + 0.2 * jax.random.normal(_keys[1], (int(nsamples / 4), 2))
        _g3 = jnp.array([-3, -3]) + 0.2 * jax.random.normal(_keys[1], (int(nsamples / 2), 2))
        # _g4 = jnp.array([3, -3]) + 0.2 * jax.random.normal(_keys[3], (int(nsamples / 4), 2))
        _xs = jnp.concatenate([_g1, _g3], axis=0)
        _ys = -5 / xs
    _keys = jax.random.split(_key, num=nsamples)
    errs = jax.vmap(compute_errs, in_axes=[0, 0, None, 0])(_xs, _ys, _param, _keys)
    return jnp.mean(errs)


@jax.jit
def opt_step_kernel(_param, _opt_state, _key):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _key)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


optimiser = optax.adam(learning_rate=1e-3)
opt_state = optimiser.init(init_param)
param = init_param

if args.train:
    for i in range(100000):
        key, subkey = jax.random.split(key)
        param, opt_state, loss = opt_step_kernel(param, opt_state, subkey)
        print(f'i: {i}, loss: {loss}')

    np.save('param_sdsb', param)
else:
    param = np.load('param_sdsb.npy')

    # Backward sampling
    def simulate_backward(xT, _key):
        def scan_body(carry, elem):
            x = carry
            t, dw = elem

            x = x + (-drift(x, xT, T - t) + nn_eval(x, T - t, param)) * dt + dw
            return x, x

        _, _subkey = jax.random.split(_key)
        dws = jnp.sqrt(dt) * jax.random.normal(_subkey, (nsteps, 2))
        return jax.lax.scan(scan_body, xT, (ts[:-1], dws[:-1]))[1]


    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=nsamples)
    backward_traj = jax.vmap(simulate_backward, in_axes=[0, 0])(ys, keys)

    fig, axes = plt.subplots()
    axes.scatter(ys[:, 0], ys[:, 1], s=1)
    axes.scatter(xs[:, 0], xs[:, 1], s=1)
    # for ax in axes:
    #     ax.set_xlim([-4, 4])
    #     ax.set_ylim([-1, 1])
    plt.show()

    fig, axes = plt.subplots()
    axes.scatter(ys[:, 0], ys[:, 1], s=1)
    axes.scatter(backward_traj[:, -1, 0], backward_traj[:, -1, 1], s=1)
    plt.tight_layout(pad=0.1)

    # Mark
    idx = 2
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=100)
    mark_y = ys[idx]
    mark_x = xs[idx]
    mark_xs = jax.vmap(simulate_backward, in_axes=[None, 0])(mark_y, keys)[:, -1, :]
    axes.scatter(mark_y[0], mark_y[1], s=10, c='tab:red')
    axes.scatter(mark_x[0], mark_x[1], s=10, c='tab:red')
    axes.scatter(mark_xs[:, 0], mark_xs[:, 1], s=10, c='black', alpha=0.5, marker='x')

    # for ax in axes:
    #     ax.set_xlim([-4, 4])
    #     ax.set_ylim([-1, 1])
    plt.show()
