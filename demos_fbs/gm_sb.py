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
from functools import partial

jax.config.update("jax_enable_x64", False)


def posterior_sampler(key_, y_):
    key_choice, key_g1, key_g2 = jax.random.split(key_, num=3)
    chol_1 = jnp.linalg.cholesky(jnp.array([[1., 0.2],
                                            [0.2, 1.]]))
    chol_2 = jnp.linalg.cholesky(jnp.array([[1., -0.2],
                                            [-0.2, 2.]]))
    g1 = y_ + chol_1 @ jax.random.normal(key_g1, (2,))
    g2 = -y_ + chol_2 @ jax.random.normal(key_g2, (2,))
    return jax.random.choice(key_choice, jnp.stack([g1, g2]), axis=0)


def joint_sampler(key_):
    key_y, key_cond = jax.random.split(key_, num=2)
    y_ = 0.1 * jax.random.normal(key_y, (1,))
    return jnp.concatenate([posterior_sampler(key_cond, y_), y_], axis=-1)


nsamples = 10000
key = jax.random.PRNGKey(666)
keys = jax.random.split(key, nsamples)

y = jnp.array(1.)
samples = jax.vmap(posterior_sampler, in_axes=[0, None])(keys, y)

plt.scatter(samples[:, 0], samples[:, 1], s=1)
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
batch_size = 128
key, subkey = jax.random.split(key)
nn_fwd = GMSBMLP(dim=3)
param_fwd, _, nn_fn_fwd = make_st_nn(subkey, nn=nn_fwd, dim_in=(3,), batch_size=batch_size)

key, subkey = jax.random.split(key)
nn_bwd = GMSBMLP(dim=3)
param_bwd, _, nn_fn_bwd = make_st_nn(subkey, nn=nn_bwd, dim_in=(3,), batch_size=batch_size)


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
schedule = optax.cosine_decay_schedule(init_value=1e-2, decay_steps=niters // 10, alpha=1e-2)
optimiser = optax.adam(learning_rate=schedule)
optimiser = optax.chain(optax.clip_by_global_norm(1.),
                        optimiser)


# Initial SB
def init_loss(param_, key_):
    key_0, key_loss = jax.random.split(key_)
    keys = jax.random.split(key_0, num=batch_size)
    z0s = jax.vmap(joint_sampler)(keys)
    fwd_fn = lambda x, k, p: F * x
    return ipf_loss_disc(param_, None, z0s, ks, Q, nn_fn_bwd, fwd_fn, key_)


@jax.jit
def optax_kernel_init(param_, opt_state, *args, **kwargs):
    loss, grad = jax.value_and_grad(init_loss)(param_, *args, **kwargs)
    updates, opt_state = optimiser.update(grad, opt_state, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state, loss


opt_state = optimiser.init(param_bwd)
# for i in range(niters):
#     key, subkey = jax.random.split(key)
#     param_bwd, opt_state, loss = optax_kernel_init(param_bwd, opt_state, subkey)
#     print(f'Iter: {i} | loss: {loss}')

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
