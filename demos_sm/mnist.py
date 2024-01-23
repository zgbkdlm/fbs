r"""
Standard score matching on MNIST
"""
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import flax.linen as nn
from fbs.data import MNIST
from fbs.sdes import make_linear_sde, make_linear_sde_score_matching_loss, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE
from fbs.nn.models import make_simple_st_nn, MNISTResConv, MNISTAutoEncoder
from fbs.nn.unet import MNISTUNet
from fbs.nn.unet_z import MNISTUNet as MNISTUNetZ
from fbs.nn import sinusoidal_embedding

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST test.')
parser.add_argument('--train', action='store_true', default=False, help='Whether train or not.')
parser.add_argument('--sde', type=str, default='const')
parser.add_argument('--nn', type=str, default='mlp')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsteps', type=int, default=50)
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--nepochs', type=int, default=20)

args = parser.parse_args()
train = args.train

print(f'Run with {train}')

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 2
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# MNIST
d = 784
key, subkey = jax.random.split(key)
dataset = MNIST(subkey, '../datasets/mnist.npz', task='deconv')


def sampler_x(key_):
    return dataset.sampler(key_)[0]


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 4)
xs = jax.vmap(sampler_x, in_axes=[0])(keys)

if not train:
    fig, axes = plt.subplots(ncols=4)
    for col in range(4):
        axes[col].imshow(xs[col].reshape(28, 28), cmap='gray')
    plt.tight_layout(pad=0.1)
    plt.show()

# Define the forward noising process which are independent OU processes
if args.sde == 'const':
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
elif args.sde == 'lin':
    sde = StationaryLinLinearSDE(a=-0.5, b=1.)
elif args.sde == 'exp':
    sde = StationaryExpLinearSDE(a=-0.5, b=1., c=1., z=1.)
else:
    raise NotImplementedError('...')
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def simulate_forward(key_, ts_):
    x0 = sampler_x(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], x0, ts_)


# Score matching
train_nsamples = args.batch_size
train_nsteps = args.nsteps
train_dt = T / train_nsteps
nepochs = args.nepochs
data_size = dataset.n
nn_param_init = nn.initializers.xavier_normal()
nn_param_dtype = jnp.float64

key, subkey = jax.random.split(key)
if args.nn == 'mlp':
    my_nn = MNISTAutoEncoder(nn_param_dtype=nn_param_dtype, nn_param_init=nn_param_init)
elif args.nn == 'unet':
    my_nn = MNISTUNet(16)
elif args.nn == 'unetz':
    my_nn = MNISTUNetZ(train_dt)
elif args.nn == 'conv':
    my_nn = MNISTResConv(dt=train_dt)
else:
    raise NotImplementedError('...')
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=d, batch_size=train_nsamples,
                                                   nn_model=my_nn)

loss_fn = make_linear_sde_score_matching_loss(sde, nn_score, t0=0., T=T, nsteps=train_nsteps, random_times=True)


@jax.jit
def optax_kernel(param_, opt_state_, key_, xy0s_):
    loss_, grad = jax.value_and_grad(loss_fn)(param_, key_, xy0s_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state_, loss_


if args.schedule == 'cos':
    schedule = optax.cosine_decay_schedule(args.lr, data_size // train_nsamples, .95)
elif args.schedule == 'exp':
    schedule = optax.exponential_decay(args.lr, data_size // train_nsamples, .95)
else:
    schedule = optax.constant_schedule(args.lr)
optimiser = optax.adam(learning_rate=schedule)
param = array_param
opt_state = optimiser.init(param)

if train:
    for i in range(nepochs):
        data_key, subkey = jax.random.split(data_key)
        perm_inds = dataset.init_enumeration(subkey, train_nsamples)
        for j in range(data_size // train_nsamples):
            subkey, subkey2 = jax.random.split(subkey)
            x0s, _ = dataset.enumerate_subset(j, perm_inds, subkey)
            param, opt_state, loss = optax_kernel(param, opt_state, subkey2, x0s)
            print(f'Epoch: {i} / {nepochs}, iter: {j} / {data_size // train_nsamples}, loss: {loss}')
        np.save(f'./mnist_{args.nn}_{args.sde}.npy', param)
else:
    param = np.load(f'./mnist_{args.nn}_{args.sde}.npy')


# Verify if the score function is learnt properly
def reverse_drift(u, t):
    return -sde.drift(u, T - t) + sde.dispersion(T - t) ** 2 * nn_score(u[None, :], T - t, param)


def reverse_dispersion(t):
    return sde.dispersion(T - t)


def backward_euler(key_, u0):
    def scan_body(carry, elem):
        u = carry
        dw, t = elem

        u = u + reverse_drift(u, t) * dt + reverse_dispersion(t) * dw
        return u, None

    _, subkey_ = jax.random.split(key_)
    dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, d))
    return jax.lax.scan(scan_body, u0, (dws, ts[:-1]))[0]


# Simulate the backward and verify if it matches the target distribution
kkk = jax.random.PRNGKey(7788)
key, subkey = jax.random.split(kkk)
test_x0 = sampler_x(subkey)
key, subkey = jax.random.split(key)
traj = simulate_cond_forward(subkey, test_x0, ts)
terminal_val = traj[-1]

fig, axes = plt.subplots(ncols=10, sharey='row')
for col in range(10):
    axes[col].imshow(traj[col * 10].reshape(28, 28), cmap='gray')
plt.tight_layout(pad=0.1)
plt.show()

key, subkey = jax.random.split(key)
approx_init_sample = backward_euler(subkey, terminal_val)
print(jnp.min(test_x0), jnp.max(test_x0))
print(jnp.min(approx_init_sample), jnp.max(approx_init_sample))

fig, axes = plt.subplots(ncols=3, sharey='row')
axes[0].imshow(test_x0.reshape(28, 28), cmap='gray')
axes[1].imshow(terminal_val.reshape(28, 28), cmap='gray')
axes[2].imshow(approx_init_sample.reshape(28, 28), cmap='gray')
plt.tight_layout(pad=0.1)
plt.show()
