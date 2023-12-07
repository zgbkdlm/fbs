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
nsamples = 10_00
jax.config.update("jax_enable_x64", True)
nn_param_init = nn.initializers.xavier_normal()
key = jax.random.PRNGKey(666)

dt = 0.01
nsteps = 100
T = nsteps * dt
ts = jnp.linspace(dt, T, nsteps)


# Neural network construction
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=20, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=5, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=1, param_dtype=jnp.float64, kernel_init=nn_param_init)(x)
        return jnp.squeeze(x)


mlp = MLP()
key, subkey = jax.random.split(key)
init_param, _, nn_eval = make_nn_with_time(mlp, dim_in=1, batch_size=10, key=subkey)


# Define forward noising model
# dx = -0.5 x dt + dw
def drift(x):
    return -0.5 * x


def dispersion(_):
    return 1.


def cond_score_t_0(x, t, x0):
    return jax.grad(jax.scipy.stats.norm.logpdf)(x, x0 * jnp.exp(-0.5 * t), jnp.sqrt(1 - jnp.exp(-t)))


def simulate_forward(x0, _key):
    def scan_body(carry, elem):
        x = carry
        rnd = elem

        x = jnp.exp(-0.5 * dt) * x + rnd
        return x, x

    _, _subkey = jax.random.split(_key)
    rnds = jnp.sqrt(1 - jnp.exp(-dt)) * jax.random.normal(_subkey, (nsteps,))
    return jax.lax.scan(scan_body, x0, rnds)[1]


# Draw some wild initial samples (e.g., Gaussian sum)
key, subkey = jax.random.split(key)
_s1, _s2 = jax.random.normal(subkey, (2, int(nsamples / 2)))
x0s = jnp.hstack([-1.5 + _s1, 1.5 + _s2])
plt.hist(x0s, density=True, bins=50, label='x0', alpha=0.5)

# Draw terminal samples
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
paths = jax.vmap(simulate_forward, in_axes=[0, 0])(x0s, keys)
xTs = paths[:, -1]
plt.hist(xTs, density=True, bins=50, label='xT', alpha=0.5)
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


# Score matching
sgm = True

if sgm:
    def loss_fn(_param, _key):
        _keys = jax.random.split(_key, num=nsamples)
        forward_paths = jax.vmap(simulate_forward, in_axes=[0, 0])(x0s, _keys)  # (nsamples, nsteps)
        errs = (jax.vmap(jax.vmap(nn_eval,
                                  in_axes=[0, 0, None]),
                         in_axes=[0, None, None])(forward_paths, ts, _param) -
                jax.vmap(jax.vmap(cond_score_t_0,
                                  in_axes=[0, 0, None]),
                         in_axes=[0, None, 0])(forward_paths, ts, x0s)) ** 2  # (nsamples, nsteps)
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

    for i in range(100):
        key, subkey = jax.random.split(key)
        param, opt_state, loss = opt_step_kernel(param, opt_state, subkey)
        print(f'i: {i}, loss: {loss}')


# Backward sampling
def simulate_backward(xT, _key):
    def scan_body(carry, elem):
        x = carry
        t, dw = elem

        if sgm:
            x = x + (-drift(x) + nn_eval(x, T - t, param)) * dt + dw
        else:
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
