"""
Try train a Schrodinger bridge between a Gaussian mixture and a unit Gaussian.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE
from fbs.dsb.base import ipf_loss_disc
from fbs.nn.models import make_st_nn, GMSBMLP
from fbs.nn.utils import make_optax_kernel
from functools import partial

jax.config.update("jax_enable_x64", False)


def posterior_sampler(key_, y_):
    key_choice, key_g1, key_g2 = jax.random.split(key_, num=3)
    std_1 = 0.5
    std_2 = 0.5
    g1 = y_ + std_1 * jax.random.normal(key_g1, (1,))
    g2 = -y_ + std_2 * jax.random.normal(key_g2, (1,))
    return jax.random.choice(key_choice, jnp.vstack([g1, g2]), axis=0)


def joint_sampler(key_):
    key_y, key_cond = jax.random.split(key_, num=2)
    y_ = 1. * jax.random.normal(key_y, (1,))
    return jnp.concatenate([posterior_sampler(key_cond, y_), y_], axis=-1)


nsamples = 10000
key = jax.random.PRNGKey(666)
keys = jax.random.split(key, nsamples)

y = jnp.array([1.])
samples = jax.vmap(posterior_sampler, in_axes=[0, None])(keys, y)

plt.hist(samples[:, 0], density=True, bins=100, alpha=0.5)
plt.show()

# SB settings
nsteps = 100
ks = jnp.arange(nsteps + 1)
T = 1.
dt = T / nsteps

sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, _, _ = make_linear_sde(sde)
F, Q = discretise_linear_sde(dt, 0.)

# NN setting
batch_size = 256
key, subkey = jax.random.split(key)
nn_fwd = GMSBMLP(dim=2)
param_fwd, _, nn_fn_fwd = make_st_nn(subkey, nn=nn_fwd, dim_in=(2,), batch_size=batch_size)

key, subkey = jax.random.split(key)
nn_bwd = GMSBMLP(dim=2)
param_bwd, _, nn_fn_bwd = make_st_nn(subkey, nn=nn_bwd, dim_in=(2,), batch_size=batch_size)


def simulate_disc(key_, z0s_, ks_, param_, fn):
    def scan_body(carry, elem):
        z = carry
        k, rnd = elem
        z = fn(z, k, param_) + jnp.sqrt(Q) * rnd
        return z, None

    n, d = z0s_.shape
    rnds = jax.random.normal(key_, (nsteps, n, d))
    return jax.lax.scan(scan_body, z0s_, (ks_[:-1], rnds))[0]


# Optax
niters = 1000
schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=niters // 100, alpha=1e-2)
# schedule = optax.constant_schedule(1e-3)
optimiser = optax.adam(learning_rate=schedule)
optimiser = optax.chain(optax.clip_by_global_norm(1.),
                        optimiser)


# Initial SB
def init_loss_fn(param_, key_):
    key_0, key_loss = jax.random.split(key_)
    keys = jax.random.split(key_0, num=batch_size)
    z0s = jax.vmap(joint_sampler)(keys)
    fwd_fn = lambda x, k, p: F * x
    return ipf_loss_disc(param_, None, z0s, ks, Q, nn_fn_bwd, fwd_fn, key_)


optax_kernel_init, ema_kernel_init = make_optax_kernel(optimiser, init_loss_fn, jit=True)

param_bwd_ema = param_bwd
opt_state = optimiser.init(param_bwd)
for i in range(niters):
    key, subkey = jax.random.split(key)
    param_bwd, opt_state, loss = optax_kernel_init(param_bwd, opt_state, subkey)
    param_bwd_ema = ema_kernel_init(param_bwd_ema, param_bwd, i, 100, 2, 0.99)
    print(f'Iter: {i} | loss: {loss}')

# Plot
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
z0s = jax.vmap(joint_sampler)(keys)

key, subkey = jax.random.split(key)
zTs = simulate_disc(subkey, z0s, ks, None, lambda x, k, p: F * x)

fig, axes = plt.subplots(ncols=2)
axes[0].scatter(z0s[:, 0], z0s[:, 1], s=1)
axes[1].scatter(zTs[:, 0], zTs[:, 1], s=1)
plt.tight_layout(pad=0.1)
plt.show()

key, subkey = jax.random.split(key)
approx_z0s = simulate_disc(subkey, zTs, ks[::-1], param_bwd, nn_fn_bwd)

fig, axes = plt.subplots(ncols=2)
axes[0].scatter(approx_z0s[:, 0], approx_z0s[:, 1], s=1)
axes[1].scatter(z0s[:, 0], z0s[:, 1], s=1)
plt.tight_layout(pad=0.1)
plt.savefig('gm_sb.png')
plt.show()
