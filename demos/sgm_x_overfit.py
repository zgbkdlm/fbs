"""
1D score matching demo, this is how they implement the score matching in practice, expect 1) random uniform time
2) time embedding .
"""
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax import linen as nn
from fbs_gauss.nn.utils import make_nn_with_time
from fbs_gauss.utils import discretise_lti_sde

# General configs
nsamples = 10_000
jax.config.update("jax_enable_x64", True)
nn_param_init = nn.initializers.xavier_normal()
key = jax.random.PRNGKey(666)

dt = 0.01
nsteps = 100
T = nsteps * dt
ts = jnp.linspace(dt, T, nsteps)


# Define forward noising model
# dx = -0.5 x dt + dw
# dy = 0
def drift(z):
    return jnp.array([-0.5 * z[0], 0.])


def simulate_forward(z0, _key):
    def scan_body(carry, elem):
        x = carry
        rnd = elem

        x = jnp.exp(-0.5 * dt) * x + rnd
        return x, x

    _, _subkey = jax.random.split(_key)
    rnds = jnp.sqrt(1 - jnp.exp(-dt)) * jax.random.normal(_subkey, (nsteps,))
    x0 = z0[0]
    xs = jax.lax.scan(scan_body, x0, rnds)[1]
    return jnp.concatenate([xs[:, 1], jnp.ones((nsteps, 1))])


# Draw initial samples
key, subkey = jax.random.split(key)
x0s = 1 + 0.1 * jax.random.normal(subkey, (nsamples,))
plt.hist(x0s, density=True, bins=50, label='x0')

# Draw terminal samples
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
paths = jax.vmap(simulate_forward, in_axes=[0, 0])(x0s, keys)
xTs = paths[:, -1]
plt.hist(xTs, density=True, bins=50, label='xT')
plt.legend()
plt.show()

# We can compute the true score to compare to that of NN
A, B = -0.5 * jnp.eye(1), jnp.eye(1)


def forward_m_var(t, m0, var0):
    F, Q = discretise_lti_sde(A, B, t)
    F = jnp.squeeze(F)
    Q = jnp.squeeze(Q)
    return F * m0, F ** 2 * var0 + Q


def true_score(x, t):
    mt, vart = forward_m_var(t, 1., 0.1 ** 2)
    return jax.grad(jax.scipy.stats.norm.logpdf, argnums=0)(x, mt, jnp.sqrt(vart))


# Backward sampling
def simulate_backward(xT, _key):
    def scan_body(carry, elem):
        x = carry
        t, dw = elem

        x = x + (-drift(x) + true_score(x, T - t)) * dt + dw
        return x, _

    _, _subkey = jax.random.split(_key)
    dws = jnp.sqrt(dt) * jax.random.normal(_subkey, (nsteps,))
    return jax.lax.scan(scan_body, xT, (ts, dws))[0]


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
approx_x0s = jax.vmap(simulate_backward, in_axes=[0, 0])(xTs, keys)
plt.hist(x0s, density=True, bins=50, label='x0')
plt.hist(approx_x0s, density=True, bins=50, label='approx x0')
plt.legend()
plt.show()
