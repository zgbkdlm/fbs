r"""
Restoration of MNIST images
"""
import argparse
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
import flax.linen as nn
from fbs.data import MNIST
from fbs.nn.models import make_simple_st_nn, MNISTConv, MNISTAutoEncoder
from fbs.filters.csmc.csmc import csmc_kernel
from fbs.filters.csmc.resamplings import killing
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST restoration.')
parser.add_argument('--train', action='store_true', help='Whether train or not.')
parser.add_argument('--nn', type=str, help='What NN structure to use.')
args = parser.parse_args()
use_pretrained = ~args.train

# General configs
nparticles = 100
nsamples = 1000
burn_in = 100
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 2
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# MNIST
d = 784 * 2
key, subkey = jax.random.split(key)
dataset = MNIST(subkey, '../datasets/mnist.npz', task='deconv')


def sampler_xy(key_):
    x_, y_ = dataset.sampler(key_)
    return jnp.concatenate([x_, y_], axis=-1)


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 4)
xys = jax.vmap(sampler_xy, in_axes=[0])(keys)

if ~use_pretrained:
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for col in range(4):
        axes[0, col].imshow(xys[col, :784].reshape(28, 28), cmap='gray')
        axes[1, col].imshow(xys[col, 784:].reshape(28, 28), cmap='gray')
    plt.tight_layout(pad=0.1)
    plt.show()

# Define the forward noising process which are independent OU processes
a = -0.5
b = 1.
gamma = b ** 2


def discretise_ou_sde(t):
    return jnp.exp(a * t), b ** 2 / (2 * a) * (jnp.exp(2 * a * t) - 1)


def cond_score_t_0(xy, t, xy0):
    F, Q = discretise_ou_sde(t)
    logpdf_ = lambda u: jnp.sum(jax.scipy.stats.norm.logpdf(u, F * xy0, Q))
    return jax.grad(logpdf_)(xy)


def score_scale(t):
    return discretise_ou_sde(t)[1]


def simulate_cond_forward(key_, xy0, ts_):
    def scan_body(carry, elem):
        xy = carry
        dt_, rnd = elem

        F, Q = discretise_ou_sde(dt_)
        xy = F * xy + jnp.sqrt(Q) * rnd
        return xy, xy

    dts = jnp.diff(ts_)
    rnds = jax.random.normal(key_, (dts.shape[0], d))
    return jnp.concatenate([xy0[None, :], jax.lax.scan(scan_body, xy0, (dts, rnds))[1]], axis=0)


def simulate_forward(key_, ts_):
    xy0 = sampler_xy(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], xy0, ts_)


# Visualise the terminal distribution
# key, subkey = jax.random.split(key)
# keys = jax.random.split(subkey, num=nsamples)
# fwd_trajs = jax.vmap(simulate_forward, in_axes=[0, None])(keys, ts)
# plt.scatter(fwd_trajs[:, -1, 0], fwd_trajs[:, -1, 1], s=1, alpha=0.5)
# plt.show()

# Score matching
train_nsamples = 100
train_nsteps = 100
nepochs = 10
data_size = dataset.n
nn_param_init = nn.initializers.xavier_normal()

if args.nn == 'conv':
    mnist_nn = MNISTConv()
elif args.nn == 'mlp':
    mnist_nn = MNISTAutoEncoder()
else:
    raise ValueError('Unknown NN structure.')
key, subkey = jax.random.split(key)
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=d, batch_size=train_nsamples,
                                                   mlp=mnist_nn)


def loss_fn(param_, key_, xy0s_):
    key_ts, key_fwd = jax.random.split(key_, num=2)
    batch_ts = jnp.hstack([0.,
                           jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0., maxval=T)),
                           T])
    batch_scale = score_scale(batch_ts[1:])
    fwd_paths = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(jax.random.split(key_fwd, num=train_nsamples),
                                                                      xy0s_, batch_ts)
    nn_evals = jax.vmap(jax.vmap(nn_score,
                                 in_axes=[0, 0, None]),
                        in_axes=[0, None, None])(fwd_paths[:, 1:], batch_ts[1:], param_)
    cond_score_evals = jax.vmap(jax.vmap(cond_score_t_0,
                                         in_axes=[0, 0, None]),
                                in_axes=[0, None, 0])(fwd_paths[:, 1:], batch_ts[1:], fwd_paths[:, 0])
    return jnp.mean(jnp.sum((nn_evals - cond_score_evals) ** 2, axis=-1) * batch_scale[None, :])


@jax.jit
def optax_kernel(param_, opt_state_, key_, xy0s_):
    loss_, grad = jax.value_and_grad(loss_fn)(param_, key_, xy0s_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state_, loss_


# schedule = optax.cosine_decay_schedule(1e-4, 20, .95)
schedule = optax.constant_schedule(1e-3)
optimiser = optax.adam(learning_rate=schedule)
param = array_param
opt_state = optimiser.init(param)

if use_pretrained:
    param = np.load(f'./mnist_{args.nn}.npy')
else:
    for i in range(nepochs):
        data_key, subkey = jax.random.split(data_key)
        perm_inds = dataset.init_enumeration(subkey, train_nsamples)
        for j in range(data_size // train_nsamples):
            subkey, subkey2 = jax.random.split(subkey)
            x0s, y0s = dataset.enumerate_subset(j, perm_inds, subkey)
            xy0s = jnp.concatenate([x0s, y0s], axis=-1)
            param, opt_state, loss = optax_kernel(param, opt_state, subkey2, xy0s)
            print(f'Epoch: {i} / {nepochs}, iter: {j} / {data_size // train_nsamples}, loss: {loss}')
    np.save(f'./mnist_{args.nn}.npy', param)


# Verify if the score function is learnt properly
def reverse_drift(uv, t):
    return -a * uv + gamma * nn_score(uv, T - t, param)


def reverse_drift_u(u, v, t):
    uv = jnp.hstack([u, v])
    return (-a * uv + gamma * nn_score(uv, T - t, param))[:784]


def reverse_drift_v(v, u, t):
    uv = jnp.hstack([u, v])
    return (-a * uv + gamma * nn_score(uv, T - t, param))[784:]


def backward_euler(key_, uv0):
    def scan_body(carry, elem):
        uv = carry
        dw, t = elem

        uv = uv + reverse_drift(uv, t) * dt + b * dw
        return uv, None

    _, subkey_ = jax.random.split(key_)
    dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, d))
    return jax.lax.scan(scan_body, uv0, (dws, ts[:-1]))[0]


# Simulate the backward and verify if it matches the target distribution
key, subkey = jax.random.split(key)
test_xy0 = sampler_xy(subkey)
key, subkey = jax.random.split(key)
terminal_val = simulate_cond_forward(subkey, test_xy0, ts)[-1]
key, subkey = jax.random.split(key)
approx_init_sample = backward_euler(subkey, terminal_val)

fig, axes = plt.subplots(nrows=2, ncols=2, sharey='row')
axes[0, 0].imshow(test_xy0[:784].reshape(28, 28))
axes[1, 0].imshow(test_xy0[784:].reshape(28, 28))
axes[0, 1].imshow(approx_init_sample[:784].reshape(28, 28))
axes[1, 1].imshow(approx_init_sample[784:].reshape(28, 28))
plt.tight_layout(pad=0.1)
plt.show()

# Now do cSMC conditional sampling
true_x0, y0 = xys[0, :784], xys[0, 784:]


def transition_sampler(us, v, t, key_):
    return (us + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us, v, t) * dt
            + math.sqrt(dt) * b * jax.random.normal(key_, us.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v, t):
    return jax.scipy.stats.multivariate_normal.logpdf(u,
                                                      u_prev + reverse_drift_u(u_prev, v, t) * dt,
                                                      math.sqrt(dt) * b)


@partial(jax.vmap, in_axes=[None, 0, None, None])
def likelihood_logpdf(v, u_prev, v_prev, t_prev):
    cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
    return jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * b)


def fwd_sampler(key_, x0):
    xy0 = jnp.hstack([x0, y0])
    return simulate_cond_forward(key_, xy0, ts)


@jax.jit
def gibbs_kernel(key_, xs_, us_star_, bs_star_):
    key_fwd, key_csmc = jax.random.split(key_)
    path_xy = fwd_sampler(key_fwd, xs_[0])
    us, vs = path_xy[::-1, :784], path_xy[::-1, 784:]

    def init_sampler(*_):
        return us[0] * jnp.ones((nparticles, us.shape[-1]))

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
xs = fwd_sampler(subkey, true_x0)[:, :784]
us_star = xs[::-1]
bs_star = jnp.zeros((nsteps + 1), dtype=int)

uss = np.zeros((nsamples, nsteps + 1, 784))
xss = np.zeros((nsamples, nsteps + 1, 784))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    xs, us_star, bs_star, acc = gibbs_kernel(subkey, xs, us_star, bs_star)
    xss[i], uss[i] = xs, us_star
    print(f'Gibbs iter: {i}')

# Plot
fig, axes = plt.subplots(ncols=10, sharey='row')
axes[0].imshow(true_x0.reshape(28, 28))
for i in range(1, 10):
    axes[i].imshow(uss[i * 10, -1].reshape(28, 28))
