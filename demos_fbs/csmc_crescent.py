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
from fbs.nn.models import make_simple_st_nn, CrescentMLP
from fbs.nn.utils import make_optax_kernel
from fbs.sdes import make_linear_sde, make_linear_sde_law_loss, StationaryLinLinearSDE, reverse_simulator
from fbs.filters.csmc.csmc import csmc_kernel
from fbs.filters.csmc.resamplings import killing
from functools import partial

# General configs
nparticles = 10
ngibbs = 2000
burn_in = 100
jax.config.update("jax_enable_x64", False)
key = jax.random.PRNGKey(666)
y0 = 4.
use_pretrained = True
use_ema = True

T = 1
nsteps = 500
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
sde = StationaryLinLinearSDE(beta_min=2e-2, beta_max=5., t0=0., T=T)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def simulate_forward(key_, ts_):
    xy0 = sampler_xy(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], xy0, ts_)


# Visualise the terminal distribution
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5000)
terminal_vals = jax.vmap(simulate_forward, in_axes=[0, None])(keys, ts)[:, -1, :]
plt.scatter(terminal_vals[:, 0], terminal_vals[:, 2], s=1, alpha=0.5)
plt.show()

# Score matching training
train_nsamples = 128
train_nsteps = 100
nn_dt = T / 200

key, subkey = jax.random.split(key)
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=3, batch_size=train_nsamples,
                                                   nn_model=CrescentMLP(nn_dt))

loss_type = 'score'
loss_fn = make_linear_sde_law_loss(sde, nn_score,
                                   t0=0., T=T, nsteps=train_nsteps,
                                   random_times=True, loss_type=loss_type)


# schedule = optax.constant_schedule(1e-3)
schedule = optax.cosine_decay_schedule(1e-3, 50, .95)
optimiser = optax.chain(optax.clip_by_global_norm(1.),
                        optax.adam(learning_rate=schedule))
optax_kernel, ema_kernel = make_optax_kernel(optimiser, loss_fn, jit=True)

param = array_param
ema_param = param
opt_state = optimiser.init(param)

if use_pretrained:
    param = np.load('./crescent.npz')['ema_param' if use_ema else 'param']
else:
    for i in range(2000):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, train_nsamples)
        samples = jax.vmap(sampler_xy, in_axes=[0])(keys)
        key, subkey = jax.random.split(key)
        param, opt_state, loss = optax_kernel(param, opt_state, subkey, samples)
        ema_param = ema_kernel(ema_param, param, i, 100, 0.99)
        print(f'i: {i}, loss: {loss}')
    np.savez('./crescent.npz', param=param, ema_param=ema_param)


# Verify if the score function is learnt properly
def rev_sim(key_, u0):
    def learnt_score(x, t):
        return nn_score(x, t, param)

    return reverse_simulator(key_, u0, ts, learnt_score, sde.drift, sde.dispersion, integrator='euler-maruyama')


# Simulate the backward and verify if it matches the target distribution
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5000)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, 0])(keys, terminal_vals)

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 5000)
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
axes[0, 0].set_xlim(-4, 4)
axes[0, 1].set_xlim(-4, 4)
axes[1, 0].set_ylim(-5, 10)
plt.tight_layout(pad=0.1)
plt.show()


def reverse_drift(uv, t):
    return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * nn_score(uv, T - t, param)


def reverse_drift_u(u, v, t):
    uv = jnp.hstack([u, v])
    return reverse_drift(uv, t)[:2]


def reverse_drift_v(v, u, t):
    uv = jnp.hstack([u, v])
    return reverse_drift(uv, t)[-1]


def reverse_dispersion(t):
    return sde.dispersion(T - t)


# Now do cSMC conditional sampling
def transition_sampler(us_prev, v_prev, t_prev, key_):
    return (us_prev + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us_prev, v_prev, t_prev) * dt
            + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                               u_prev + reverse_drift_u(u_prev, v_prev, t_prev) * dt,
                                               math.sqrt(dt) * reverse_dispersion(t_prev)))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def likelihood_logpdf(v, u_prev, v_prev, t_prev):
    return jax.scipy.stats.norm.logpdf(v,
                                       v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt,
                                       math.sqrt(dt) * reverse_dispersion(t_prev))


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

uss = np.zeros((ngibbs, nsteps + 1, 2))
xss = np.zeros((ngibbs, nsteps + 1, 2))
for i in range(ngibbs):
    key, subkey = jax.random.split(key)
    xs, us_star, bs_star, acc = gibbs_kernel(subkey, xs, us_star, bs_star)
    xss[i], uss[i] = xs, us_star
    print(f'Gibbs iter: {i}, acc: {jnp.mean(acc)}')

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

# Check the joint in terms of 2D hist
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
pcm = axes[0].pcolormesh(lines_, lines_, post)
axes[1].hist2d(uss[:, -1, 0], uss[:, -1, 1], bins=100, density=True)
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)
axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-4, 4)
fig.colorbar(pcm, ax=axes[0])
plt.tight_layout(pad=0.1)
plt.show()

# Check the marginal
plt.plot(lines_, jax.scipy.integrate.trapezoid(post, lines_, axis=1))
plt.hist(uss[:, -1, 1], bins=50, density=True, alpha=0.5)
plt.show()

plt.plot(lines_, jax.scipy.integrate.trapezoid(post, lines_, axis=0))
plt.hist(uss[:, -1, 0], bins=50, density=True, alpha=0.5)
plt.show()
