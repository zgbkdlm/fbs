r"""
Conditional sampling on MNIST
"""
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fbs.data import MNIST
from fbs.sdes import make_linear_sde, make_linear_sde_law_loss, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE, reverse_simulator
from fbs.nn.models import make_simple_st_nn
from fbs.nn.unet_z import MNISTUNet
from fbs.nn.utils import make_optax_kernel

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST test.')
parser.add_argument('--train', action='store_true', default=False, help='Whether train or not.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--upsampling', type=str, default='pixel_shuffle')
parser.add_argument('--loss_type', type=str, default='score')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--nsteps', type=int, default=2)
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--nepochs', type=int, default=30)
parser.add_argument('--grad_clip', action='store_true', default=False)
parser.add_argument('--test_nsteps', type=int, default=500)
parser.add_argument('--test_epoch', type=int, default=12)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)

args = parser.parse_args()
train = args.train

print(f'Run with {train}')

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 1
nsteps = args.test_nsteps
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# MNIST
d = 784 * 2
key, subkey = jax.random.split(key)
dataset = MNIST(subkey, '../datasets/mnist.npz', task='deconv')


def sampler(key_):
    x_, y_ = dataset.sampler(key_)
    return dataset.concat(x_, y_)


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 4)
xys = jax.vmap(sampler, in_axes=[0])(keys)

if not train:
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for row in range(2):
        for col in range(4):
            axes[row, col].imshow(xys[col, row].reshape(28, 28), cmap='gray')
    plt.tight_layout(pad=0.1)
    plt.show()

# Define the forward noising process which are independent OU processes
if args.sde == 'const':
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
elif args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=5., t0=0., T=T)
elif args.sde == 'exp':
    sde = StationaryExpLinearSDE(a=-0.5, b=1., c=1., z=1.)
else:
    raise NotImplementedError('...')
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def simulate_forward(key_, ts_):
    xy0 = sampler(key_)
    return simulate_cond_forward(jax.random.split(key_)[1], xy0, ts_)


# Score matching
train_nsamples = args.batch_size
train_nsteps = args.nsteps
nn_dt = T / 200
nepochs = args.nepochs
data_size = dataset.n

key, subkey = jax.random.split(key)
my_nn = MNISTUNet(dt=nn_dt, upsampling_method=args.upsampling)
_, _, array_param, _, nn_score = make_simple_st_nn(subkey,
                                                   dim_in=d, batch_size=train_nsamples,
                                                   nn_model=my_nn)

loss_type = args.loss_type
loss_fn = make_linear_sde_law_loss(sde, nn_score, t0=0., T=T, nsteps=train_nsteps,
                                   random_times=True, loss_type=loss_type)


if args.schedule == 'cos':
    schedule = optax.cosine_decay_schedule(args.lr, data_size // train_nsamples * 4, .95)
elif args.schedule == 'exp':
    schedule = optax.exponential_decay(args.lr, data_size // train_nsamples * 4, .95)
else:
    schedule = optax.constant_schedule(args.lr)

if args.grad_clip:
    optimiser = optax.adam(learning_rate=schedule)
    optimiser = optax.chain(optax.clip_by_global_norm(1.),
                            optimiser)
else:
    optimiser = optax.adam(learning_rate=schedule)

optax_kernel, ema_kernel = make_optax_kernel(optimiser, loss_fn, jit=True)
param = array_param
ema_param = param
opt_state = optimiser.init(param)

if train:
    for i in range(nepochs):
        data_key, subkey = jax.random.split(data_key)
        perm_inds = dataset.init_enumeration(subkey, train_nsamples)
        for j in range(data_size // train_nsamples):
            subkey, subkey2 = jax.random.split(subkey)
            x0s, y0s = dataset.enumerate_subset(j, perm_inds, subkey)
            xy0s = dataset.concat(x0s, y0s)
            param, opt_state, loss = optax_kernel(param, opt_state, subkey2, xy0s)
            ema_param = ema_kernel(ema_param, param, j, 200, 0.99)
            print(f'| {args.upsampling} | {args.sde} | {loss_type} | {args.schedule} | '
                  f'Epoch: {i} / {nepochs}, iter: {j} / {data_size // train_nsamples}, loss: {loss:.4f}')
        np.savez(f'./mnist_{args.upsampling}_{args.sde}_'
                 f'{loss_type}_{args.schedule}_{"clip_" if args.grad_clip else ""}{i}.npz',
                 param=param, ema_param=ema_param)
else:
    param = np.load(f'./mnist_{args.upsampling}_{args.sde}_{loss_type}_{args.schedule}_'
                    f'{"clip_" if args.grad_clip else ""}'
                    f'{args.test_epoch}.npz')['ema_param' if args.test_ema else 'param']


# Verify if the score function is learnt properly
def rev_sim(key_, u0):
    def learnt_score(x, t):
        return nn_score(x, t, param)

    return reverse_simulator(key_, u0, ts, learnt_score, sde.drift, sde.dispersion, integrator='euler-maruyama')


# Simulate the backward and verify if it matches the target distribution
kkk = jax.random.PRNGKey(args.test_seed)
key, subkey = jax.random.split(kkk)
test_x0 = sampler(subkey)
key, subkey = jax.random.split(key)
traj = simulate_cond_forward(subkey, test_x0, ts)
terminal_val = traj[-1]

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, None])(keys, terminal_val)
print(jnp.min(approx_init_samples), jnp.max(approx_init_samples))

fig, axes = plt.subplots(ncols=7, sharey='row')
axes[0].imshow(test_x0.reshape(28, 28), cmap='gray')
axes[1].imshow(terminal_val.reshape(28, 28), cmap='gray')
for i in range(2, 7):
    axes[i].imshow(approx_init_samples[i - 2].reshape(28, 28), cmap='gray')
plt.tight_layout(pad=0.1)
plt.show()
