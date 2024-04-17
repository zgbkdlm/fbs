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


def data_sampler(key_):
    key_y, key_cond = jax.random.split(key_, num=2)
    y_ = 1. * jax.random.normal(key_y, (1,))
    return jnp.concatenate([posterior_sampler(key_cond, y_), y_], axis=-1)


def ref_sampler(key_):
    return jax.random.normal(key_, (2,))


nsamples = 10000
key = jax.random.PRNGKey(666)
keys = jax.random.split(key, nsamples)

y = jnp.array([1.])
samples = jax.vmap(posterior_sampler, in_axes=[0, None])(keys, y)

plt.hist(samples[:, 0], density=True, bins=100, alpha=0.5)
plt.show()

# SB settings
nsbs = 10  # number of SB iterations
nsteps = 20
ks = jnp.arange(nsteps + 1)
T = 0.5
dt = T / nsteps

sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, _, _ = make_linear_sde(sde)
F, Q = discretise_linear_sde(dt, 0.)
print(F, Q)

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


# Optax setting
niters = 2000
# schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=niters // 10)
schedule = optax.constant_schedule(1e-3)
# schedule = optax.exponential_decay(1e-2, niters // 100, .96)
optimiser = optax.adam(learning_rate=schedule)


# optimiser = optax.chain(optax.clip_by_global_norm(1.),
#                         optimiser)

def bwd_loss_fn(param_bwd_, param_fwd_, fwd_fn, bwd_fn, key_):
    """Simulate the forward data -> sth. to learn its backward.
    """
    key_data, key_loss = jax.random.split(key_)
    keys_ = jax.random.split(key_data, num=batch_size)
    data_samples = jax.vmap(data_sampler)(keys_)
    return ipf_loss_disc(param_bwd_, param_fwd_, data_samples, ks, Q, bwd_fn, fwd_fn, key_loss)


def fwd_loss_fn(param_fwd_, param_bwd_, fwd_fn, bwd_fn, key_):
    """Simulate the backward sth. <- ref to learn its forward.
    """
    key_ref, key_loss = jax.random.split(key_)
    keys_ = jax.random.split(key_ref, num=batch_size)
    ref_samples = jax.vmap(ref_sampler)(keys_)
    return ipf_loss_disc(param_fwd_, param_bwd_, ref_samples, ks[::-1], Q, fwd_fn, bwd_fn, key_loss)


optax_kernel_bwd, _ = make_optax_kernel(optimiser, bwd_loss_fn, jit=False)
optax_kernel_fwd, _ = make_optax_kernel(optimiser, fwd_loss_fn, jit=False)
optax_kernel_bwd = jax.jit(optax_kernel_bwd, static_argnums=[3, 4])
optax_kernel_fwd = jax.jit(optax_kernel_fwd, static_argnums=[3, 4])


def sb_kernel(param_fwd_, param_bwd_, fwd_fn, bwd_fn, key_, sb_step):
    # Compute the backward
    opt_state = optimiser.init(param_bwd_)
    for i in range(niters):
        key_, subkey_ = jax.random.split(key_)
        param_bwd_, opt_state, loss = optax_kernel_bwd(param_bwd_, opt_state, param_fwd_, fwd_fn, bwd_fn, subkey_)
        print(f'Learning backward | SB: {sb_step} | iter: {i} | loss: {loss}')

    # Compute the forward
    opt_state = optimiser.init(param_fwd_)
    for i in range(niters):
        key_, subkey_ = jax.random.split(key_)
        param_fwd_, opt_state, loss = optax_kernel_fwd(param_fwd_, opt_state, param_bwd_, fwd_fn, bwd_fn, subkey_)
        print(f'Learning forward | SB: {sb_step} | iter: {i} | loss: {loss}')

    return param_fwd_, param_bwd_


# Init SB step
key, subkey = jax.random.split(key)
param_fwd, param_bwd = sb_kernel(param_fwd, param_bwd, lambda x, k, p: F * x, nn_fn_bwd, subkey, 0)

# SB iterations
for j in range(1, nsbs):
    key, subkey = jax.random.split(key)
    param_fwd, param_bwd = sb_kernel(param_fwd, param_bwd, nn_fn_fwd, nn_fn_bwd, subkey, j)

# Plot
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, nsamples)
data_samples = jax.vmap(data_sampler)(keys)

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, nsamples)
ref_samples = jax.vmap(ref_sampler)(keys)

key, subkey = jax.random.split(key)
approx_ref_samples = simulate_disc(subkey, data_samples, ks, param_fwd, nn_fn_fwd)

key, subkey = jax.random.split(key)
approx_data_samples = simulate_disc(subkey, ref_samples, ks[::-1], param_bwd, nn_fn_bwd)

fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].scatter(data_samples[:, 0], data_samples[:, 1], s=1, alpha=0.7)
axes[0, 1].scatter(approx_data_samples[:, 0], approx_data_samples[:, 1], s=1, alpha=0.7)

axes[1, 0].scatter(ref_samples[:, 0], ref_samples[:, 1], s=1, alpha=0.7)
axes[1, 1].scatter(approx_ref_samples[:, 0], approx_ref_samples[:, 1], s=1, alpha=0.7)

plt.tight_layout(pad=0.1)
plt.show()