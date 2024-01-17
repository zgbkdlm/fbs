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
from fbs.sdes import make_ou_sde, make_ou_score_matching_loss
from fbs.nn.models import make_simple_st_nn
from fbs.nn.unet import MNISTUNet
from fbs.nn import sinusoidal_embedding

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST test.')
parser.add_argument('--train', action='store_true', default=False, help='Whether train or not.')
parser.add_argument('--nn', type=str, default='mlp')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--nepochs', type=int, default=20)
args = parser.parse_args()
train = args.train

print(f'Run with {train}')

# General configs
jax.config.update("jax_enable_x64", True)
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
a = -0.5
b = 1.
gamma = b ** 2

discretise_ou_sde, cond_score_t_0, simulate_cond_forward = make_ou_sde(a, b)


def simulate_forward(key_, ts_):
    x0 = sampler_x(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], x0, ts_)


# Score matching
train_nsamples = args.batch_size
train_nsteps = 100
train_dt = T / train_nsteps
nepochs = args.nepochs
data_size = dataset.n
nn_param_init = nn.initializers.xavier_normal()
nn_param_dtype = jnp.float64


class MNISTAutoEncoder(nn.Module):
    @nn.compact
    def __call__(self, x, t):
        x = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)

        t = sinusoidal_embedding(t / train_dt, out_dim=128, max_period=train_nsteps)
        t = nn.Dense(features=32, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)

        # t = sinusoidal_embedding(t, out_dim=32)

        z = jnp.concatenate([x, t], axis=-1)
        z = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        z = nn.gelu(z)
        z = nn.Dense(features=784, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(z)
        return jnp.squeeze(z)

# class MNISTAutoEncoder(nn.Module):
#     @nn.compact
#     def __call__(self, x, t):
#         t = sinusoidal_embedding(t / train_dt, out_dim=16, max_period=train_nsteps)
#         t = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
#         t = nn.relu(t)
#         t = nn.Dense(features=256, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
#
#         x = jnp.concatenate([x, t], axis=-1)
#         x = nn.Dense(features=256, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
#         x = nn.relu(x)
#         x_ = x
#         x = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
#         x = nn.relu(x)
#
#         x = nn.Dense(features=256, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
#         x = nn.relu(x) + x_
#
#         x = nn.Dense(features=784, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(x)
#         return jnp.squeeze(x)

# class MNISTConv(nn.Module):
#     @nn.compact
#     def __call__(self, x, t):
#         x = x.reshape(-1, 28, 28, 1)
#         batch_size = x.shape[0]
#         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x1 = x
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#         x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x2 = x
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#
#         t = sinusoidal_embedding(t, out_dim=32)
#         t = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
#         t = t.reshape(batch_size, 1, 1, -1)
#
#         t1, t2 = t[:, :, :, :64], t[:, :, :, 64:]
#
#         x = x * t1 + t2
#         x = jax.image.resize(x, (x.shape[0], 14, 14, 64), 'bilinear')
#         x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = x + x2
#         x = jax.image.resize(x, (x.shape[0], 28, 28, 64), 'bilinear')
#         x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#         x = nn.relu(x)
#         x = x + x1
#         x = nn.Conv(features=1, kernel_size=(3, 3))(x)
#
#         x = x.reshape((x.shape[0], -1))
#         return jnp.squeeze(x)

class MNISTConv(nn.Module):

    @nn.compact
    def __call__(self, x, t):
        x = x.reshape(-1, 28, 28, 1)

        t = sinusoidal_embedding(t / train_dt, out_dim=32)
        t = nn.Dense(features=128, param_dtype=nn_param_dtype, kernel_init=nn_param_init)(t)
        t = nn.relu(t)
        t = t.reshape(x.shape[0], 1, 1, -1)
        t1, t2 = t[:, :, :, :64], t[:, :, :, 64:]

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(2, 2))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x * t1 + t2

        x = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=16, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.Conv(features=1, kernel_size=(3, 3))(x)

        x = x.reshape((x.shape[0], -1))

        return jnp.squeeze(x)


key, subkey = jax.random.split(key)
if args.nn == 'mlp':
    my_nn = MNISTAutoEncoder()
elif args.nn == 'unet':
    my_nn = MNISTUNet(32)
elif args.nn == 'conv':
    my_nn = MNISTConv()
else:
    raise NotImplementedError('...')
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=d, batch_size=train_nsamples,
                                                   mlp=my_nn)

loss_fn = make_ou_score_matching_loss(a, b, nn_score, t0=0., T=T, nsteps=train_nsteps, random_times=True)


@jax.jit
def optax_kernel(param_, opt_state_, key_, xy0s_):
    loss_, grad = jax.value_and_grad(loss_fn)(param_, key_, xy0s_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_)
    param_ = optax.apply_updates(param_, updates)
    return param_, opt_state_, loss_


if args.schedule == 'cos':
    schedule = optax.cosine_decay_schedule(args.lr, data_size // train_nsamples, .91)
elif args.schedule == 'exp':
    schedule = optax.exponential_decay(args.lr, data_size // train_nsamples, .91)
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
        np.save(f'./mnist_{args.nn}_{args.schedule}_{args.lr}.npy', param)
else:
    param = np.load(f'./mnist_{args.nn}_{args.schedule}_{args.lr}.npy')


# Verify if the score function is learnt properly
def reverse_drift(u, t):
    return -a * u + gamma * nn_score(u, T - t, param)


def backward_euler(key_, u0):
    def scan_body(carry, elem):
        u = carry
        dw, t = elem

        u = u + reverse_drift(u, t) * dt + b * dw
        return u, None

    _, subkey_ = jax.random.split(key_)
    dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, d))
    return jax.lax.scan(scan_body, u0, (dws, ts[:-1]))[0]


# Simulate the backward and verify if it matches the target distribution
kkk = jax.random.PRNGKey(111)
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

fig, axes = plt.subplots(ncols=3, sharey='row')
axes[0].imshow(test_x0.reshape(28, 28), cmap='gray')
axes[1].imshow(terminal_val.reshape(28, 28), cmap='gray')
axes[2].imshow(approx_init_sample.reshape(28, 28), cmap='gray')
plt.tight_layout(pad=0.1)
plt.show()
