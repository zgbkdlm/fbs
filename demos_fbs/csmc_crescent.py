r"""
Conditional sampling on a 2D crescent distribution.

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
from fbs.data import Crescent
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
y0 = 3.
use_pretrained = True

T = 3
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

crescent = Crescent()


def sampler_xy(key_):
    x_, y_ = crescent.sampler(key_, 1)
    return jnp.hstack([x_[0], y_[0]])


lines_ = jnp.linspace(-5, 5, 1000)
mesh = jnp.dstack(jnp.meshgrid(lines_, lines_))
post = crescent.posterior(mesh, y0)

plt.pcolormesh(lines_, lines_, post)
plt.legend()
plt.show()

# Define the forward noising process
A = -0.5 * jnp.eye(3)
B = jnp.eye(3)  # This code applies to diagonal B only
gamma = B @ B.T
chol_gamma = jnp.linalg.cholesky(gamma)


def cond_score_t_0(xy, t, xy0):
    F, Q = discretise_lti_sde(A, gamma, t)
    return jax.grad(jax.scipy.stats.multivariate_normal.logpdf)(xy, F @ xy0, Q)


def score_scale(t):
    variance = 1 ** 2 / (2 * -0.5) * (jnp.exp(2 * -0.5 * t) - 1)
    return variance


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


# Visualise the terminal distribution
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
fwd_trajs = jax.vmap(simulate_forward, in_axes=[0, None])(keys, ts)
plt.scatter(fwd_trajs[:, -1, 0], fwd_trajs[:, -1, 2], s=1, alpha=0.5)
plt.show()

# Score matching training
batch_nsamples = 200
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
        t = sinusoidal_embedding(t, out_dim=32)
        t = nn.Dense(features=16, param_dtype=jnp.float64, kernel_init=nn_param_init)(t)
        t = nn.relu(t)
        t = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(t)

        z = jnp.concatenate([xy, t], axis=-1)
        z = nn.Dense(features=16, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=8, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
        z = nn.relu(z)
        z = nn.Dense(features=3, param_dtype=jnp.float64, kernel_init=nn_param_init)(z)
        return jnp.squeeze(z)


key, subkey = jax.random.split(key)
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=3, batch_size=batch_nsamples,
                                                   mlp=MLP())


def loss_fn(param_, key_):
    key_ts, key_fwd = jax.random.split(key_, num=2)
    batch_ts = jnp.hstack([0.,
                           jnp.sort(jax.random.uniform(key_ts, (batch_nsteps - 1,), minval=0., maxval=T)),
                           T])
    # batch_ts = ts
    batch_scale = score_scale(batch_ts[1:])
    fwd_paths = jax.vmap(simulate_forward, in_axes=[0, None])(jax.random.split(key_fwd, num=batch_nsamples),
                                                              batch_ts)
    nn_evals = jax.vmap(jax.vmap(nn_score,
                                 in_axes=[0, 0, None]),
                        in_axes=[0, None, None])(fwd_paths[:, 1:], batch_ts[1:], param_)
    cond_score_evals = jax.vmap(jax.vmap(cond_score_t_0,
                                         in_axes=[0, 0, None]),
                                in_axes=[0, None, 0])(fwd_paths[:, 1:], batch_ts[1:], fwd_paths[:, 0])
    # cond_score_evals = jax.vmap(jax.vmap(cond_score_t_0,
    #                                      in_axes=[0, 0, 0]),
    #                             in_axes=[0, None, 0])(fwd_paths[:, 1:], jnp.diff(batch_ts), fwd_paths[:, :-1])
    return jnp.mean(jnp.sum((nn_evals - cond_score_evals) ** 2, axis=-1) * batch_scale[None, :])


@jax.jit
def optax_kernel(param_, opt_state_, key_):
    loss_, grad = jax.value_and_grad(loss_fn)(param_, key_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state_, loss_


schedule = optax.cosine_decay_schedule(1e-2, 10, .95)
# schedule = optax.constant_schedule(1e-2)
optimiser = optax.adam(learning_rate=schedule)
param = array_param
opt_state = optimiser.init(param)

if use_pretrained:
    param = np.load('./crescent.npy')
else:
    for i in range(ntrains):
        key, subkey = jax.random.split(key)
        param, opt_state, loss = optax_kernel(param, opt_state, subkey)
        print(f'i: {i}, loss: {loss}')
    np.save('./crescent.npy', param)


# Verify if the score function is learnt properly
def reverse_drift(uv, t):
    return -A @ uv + gamma @ nn_score(uv, T - t, param)


def reverse_drift_u(u, v, t):
    uv = jnp.hstack([u, v])
    return (-A @ uv + gamma @ nn_score(uv, T - t, param))[:2]


def reverse_drift_v(v, u, t):
    uv = jnp.hstack([u, v])
    return (-A @ uv + gamma @ nn_score(uv, T - t, param))[-1]


def backward_euler(key_, uv0):
    def scan_body(carry, elem):
        uv = carry
        dw, t = elem

        uv = uv + reverse_drift(uv, t) * dt + B @ dw
        return uv, None

    _, subkey_ = jax.random.split(key_)
    dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, 3))
    return jax.lax.scan(scan_body, uv0, (dws, ts[:-1]))[0]


# Simulate the backward and verify if it matches the target distribution
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
approx_init_samples = jax.vmap(backward_euler, in_axes=[0, 0])(keys, fwd_trajs[:, -1])

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, nsamples)
xys = jax.vmap(sampler_xy, in_axes=[0])(keys)

fig, axes = plt.subplots(nrows=3, ncols=2, sharey='row', sharex='col')
axes[0, 0].scatter(xys[:, 0], xys[:, 1], s=1, alpha=0.5, label='True p(x0, x1)')
axes[0, 1].scatter(approx_init_samples[:, 0], approx_init_samples[:, 1], s=1, alpha=0.5, label='Approx. p(x0, x1)')
axes[0, 0].legend()
axes[0, 1].legend()
axes[1, 0].scatter(xys[:, 0], xys[:, 2], s=1, alpha=0.5, label='True p(x0, y)')
axes[1, 1].scatter(approx_init_samples[:, 0], approx_init_samples[:, 2], s=1, alpha=0.5, label='Approx. p(x0, y)')
axes[1, 0].legend()
axes[1, 1].legend()
axes[2, 0].scatter(xys[:, 1], xys[:, 2], s=1, alpha=0.5, label='True p(x1, y)')
axes[2, 1].scatter(approx_init_samples[:, 1], approx_init_samples[:, 2], s=1, alpha=0.5, label='Approx. p(x1, y)')
axes[2, 0].legend()
axes[2, 1].legend()
plt.tight_layout(pad=0.1)
plt.show()


# Now do cSMC conditional sampling
def transition_sampler(us, v, t, key_):
    return (us + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us, v, t) * dt
            + math.sqrt(dt) * jax.random.normal(key_, us.shape) @ B[:2, :2])


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v, t):
    return jax.scipy.stats.multivariate_normal.logpdf(u,
                                                      u_prev + reverse_drift_u(u_prev, v, t) * dt,
                                                      dt * B[:2, :2] ** 2)


@partial(jax.vmap, in_axes=[None, 0, None, None])
def likelihood_logpdf(v, u_prev, v_prev, t_prev):
    cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
    return jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * B[-1, -1])


def fwd_sampler(key_, x0):
    xy0 = jnp.hstack([x0, y0])
    return simulate_cond_forward(key_, xy0, ts)


@jax.jit
def gibbs_kernel(key_, xs_, us_star_, bs_star_):
    key_fwd, key_csmc = jax.random.split(key_)
    path_xy = fwd_sampler(key_fwd, xs_[0])
    us, vs = path_xy[::-1, :2], path_xy[::-1, -1]

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
xs = fwd_sampler(subkey, jnp.zeros((2,)))[:, :2]
us_star = xs[::-1]
bs_star = jnp.zeros((nsteps + 1), dtype=int)

uss = np.zeros((nsamples, nsteps + 1, 2))
xss = np.zeros((nsamples, nsteps + 1, 2))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    xs, us_star, bs_star, acc = gibbs_kernel(subkey, xs, us_star, bs_star)
    xss[i], uss[i] = xs, us_star
    print(f'Gibbs iter: {i}')

# Plot
plt.plot(uss[:, -1, 0])
plt.plot(uss[:, -1, 1])
plt.show()

uss = uss[burn_in:]

# Check the joint
fig, ax = plt.subplots()
pcm = ax.pcolormesh(lines_, lines_, post)
ax.scatter(uss[:, -1, 0], uss[:, -1, 1], c='tab:red', s=1, alpha=0.5, label=f'Approx. p(x | y = {y0})')
ax.legend()
fig.colorbar(pcm, ax=ax)
plt.tight_layout(pad=0.1)
plt.show()

# Check the marginal
plt.plot(lines_, jax.scipy.integrate.trapezoid(post, lines_, axis=1))
plt.hist(uss[:, -1, 1], bins=50, density=True, alpha=0.5)
plt.show()

plt.plot(lines_, jax.scipy.integrate.trapezoid(post, lines_, axis=0))
plt.hist(uss[:, -1, 0], bins=50, density=True, alpha=0.5)
plt.show()
