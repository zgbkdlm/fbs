r"""
Standard score matching on crescent
"""
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import flax.linen as nn
from fbs.data import Crescent
from fbs.sdes import make_linear_sde, make_linear_sde_law_loss, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE, reverse_simulator, discrete_time_simulator
from fbs.nn.models import make_simple_st_nn, CrescentMLP
from fbs.nn import sinusoidal_embedding

# Parse arguments
parser = argparse.ArgumentParser(description='Crescent test.')
parser.add_argument('--train', action='store_true', default=True, help='Whether train or not.')
parser.add_argument('--nn', type=str, default='mlp')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--schedule', type=str, default='const')
parser.add_argument('--nepochs', type=int, default=30)
args = parser.parse_args()
train = args.train

print(f'Run with {train}')

# General configs
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 1
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)
test_nsamples = 10000

# Crescent
crescent = Crescent()


def sampler_x(key_):
    x_, y_ = crescent.sampler(key_, 1)
    return jnp.hstack([x_[0], y_[0]])


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, test_nsamples)
xs = jax.vmap(sampler_x, in_axes=[0])(keys)

fig, axes = plt.subplots(nrows=3, sharey='row', sharex='col')
axes[0].scatter(xs[:, 0], xs[:, 1], s=1, alpha=0.5, label='True p(x0, x1)')
axes[0].legend()
axes[1].scatter(xs[:, 0], xs[:, 2], s=1, alpha=0.5, label='True p(x0, y)')
axes[1].legend()
axes[2].scatter(xs[:, 1], xs[:, 2], s=1, alpha=0.5, label='True p(x1, y)')
axes[2].legend()
plt.tight_layout(pad=0.1)
plt.show()

# Define the forward noising process which are independent OU processes
# sde = StationaryExpLinearSDE(a=-0.5, b=1., c=1., z=1.)
# sde = StationaryConstLinearSDE(a=-0.5, b=1.)
sde = StationaryLinLinearSDE(beta_min=1e-2, beta_max=3., t0=0., T=T)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def simulate_forward(key_, ts_):
    x0 = sampler_x(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], x0, ts_)


# Score matching
train_nsamples = 128
train_nsteps = 100
train_dt = T / train_nsteps
nn_param_init = nn.initializers.xavier_normal()
nn_param_dtype = jnp.float64

key, subkey = jax.random.split(key)
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=3, batch_size=train_nsamples,
                                                   nn_model=CrescentMLP(train_dt))

loss_type = 'score'
loss_fn = make_linear_sde_law_loss(sde, nn_score,
                                   t0=0., T=T, nsteps=train_nsteps,
                                   random_times=True, loss_type=loss_type)


@jax.jit
def optax_kernel(param_, opt_state_, key_, xy0s_):
    loss_, grad = jax.value_and_grad(loss_fn)(param_, key_, xy0s_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state_, loss_


if args.schedule == 'cos':
    schedule = optax.cosine_decay_schedule(args.lr, 50, .95)
elif args.schedule == 'exp':
    schedule = optax.exponential_decay(args.lr, 50, .95)
else:
    schedule = optax.constant_schedule(args.lr)
optimiser = optax.adam(learning_rate=schedule)
# optimiser = optax.chain(optimiser,
#                         optax.ema(decay=0.99))
param = array_param
opt_state = optimiser.init(param)

if not train:
    param = np.load('./crescent.npy')
else:
    for i in range(1000):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, train_nsamples)
        samples = jax.vmap(sampler_x, in_axes=[0])(keys)
        key, subkey = jax.random.split(key)
        param, opt_state, loss = optax_kernel(param, opt_state, subkey, samples)
        print(f'i: {i}, loss: {loss}')
    np.save('./crescent.npy', param)

# Verify if the score function is learnt properly
if 'score' in loss_type:
    def learnt_score(x, t):
        return nn_score(x, t, param)


    def rev_sim(key_, u0):
        return reverse_simulator(key_, u0, ts, learnt_score, sde.drift, sde.dispersion, integrator='euler-maruyama')
elif loss_type == 'ipf':
    def rev_sim(key_, u0):
        return discrete_time_simulator(key_, u0, ts,
                                       lambda x, t, _: nn_score(x, T - t, param),
                                       lambda t, t_prev: jnp.sqrt(discretise_linear_sde(T - t_prev, T - t)[1]))
else:
    raise NotImplementedError(f'Loss {loss_type} not implemented.')

# Simulate the backward and verify if it matches the target distribution
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, test_nsamples)
test_x0s = jax.vmap(sampler_x)(keys)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, test_nsamples)
traj = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(keys, test_x0s, ts)
terminal_vals = traj[:, -1, :]

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, test_nsamples)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, 0])(keys, terminal_vals)

fig, axes = plt.subplots(nrows=3, ncols=2, sharey='row', sharex='col')
axes[0, 0].scatter(test_x0s[:, 0], test_x0s[:, 1], s=1, alpha=0.5, label='True p(x0, x1)')
axes[0, 1].scatter(approx_init_samples[:, 0], approx_init_samples[:, 1], s=1, alpha=0.5, label='Approx. p(x0, x1)')
axes[0, 0].legend()
axes[0, 1].legend()
axes[1, 0].scatter(test_x0s[:, 0], test_x0s[:, 2], s=1, alpha=0.5, label='True p(x0, y)')
axes[1, 1].scatter(approx_init_samples[:, 0], approx_init_samples[:, 2], s=1, alpha=0.5, label='Approx. p(x0, y)')
axes[1, 0].legend()
axes[1, 1].legend()
axes[2, 0].scatter(test_x0s[:, 1], test_x0s[:, 2], s=1, alpha=0.5, label='True p(x1, y)')
axes[2, 1].scatter(approx_init_samples[:, 1], approx_init_samples[:, 2], s=1, alpha=0.5, label='Approx. p(x1, y)')
axes[2, 0].legend()
axes[2, 1].legend()
plt.tight_layout(pad=0.1)
plt.show()
