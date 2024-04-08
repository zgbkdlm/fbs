r"""
Training forward noising model.

Run the script under the folder `./experiments`.
"""
import math
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fbs.data import CelebAHQRestore, MNISTRestore
from fbs.data.images import normalise
from fbs.sdes import make_linear_sde, make_linear_sde_law_loss, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE, reverse_simulator
from fbs.samplers import gibbs_init, gibbs_kernel
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet
from fbs.nn.utils import make_optax_kernel
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='Training forward noising model.')
parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset. Options are mnist, celeba-64, '
                                                                 'or celeba-128.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--upsampling', type=str, default='pixel_shuffle')
parser.add_argument('--loss_type', type=str, default='score')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--nsteps', type=int, default=2)
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--nepochs', type=int, default=40)
parser.add_argument('--save_mem', action='store_true', default=False,
                    help='Save memory by sharing the batch of x and t')
parser.add_argument('--grad_clip', action='store_true', default=False)

args = parser.parse_args()
dataset_name = args.dataset
resolution = 28 if dataset_name == 'mnist' else int(dataset_name.split('-')[-1])

print(f'Train on {args.dataset}')

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 2
nsteps = 500
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# Load dataset
key, subkey = jax.random.split(key)
if dataset_name == 'mnist':
    d = (resolution, resolution, 1)
    dataset = MNISTRestore(subkey, '../datasets/mnist.npz', task='inpaint-15')
elif 'celeba' in dataset_name:
    d = (resolution, resolution, 3)
    dataset = CelebAHQRestore(subkey, f'datasets/celeba_hq{resolution}.npy', task='inpaint-15', resolution=resolution)
else:
    raise NotImplementedError(f'{dataset_name} not implemented.')

# Define the forward noising process
if args.sde == 'const':
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
elif args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=5., t0=0., T=T)
elif args.sde == 'exp':
    # Not tested.
    sde = StationaryExpLinearSDE(a=-0.5, b=1., c=1., z=1.)
else:
    raise NotImplementedError('...')
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

# Score matching
train_nsamples = args.batch_size
train_nsteps = args.nsteps
nn_dt = T / 200
nepochs = args.nepochs
data_size = dataset.n

key, subkey = jax.random.split(key)
my_nn = UNet(dt=nn_dt, dim=64, upsampling=args.upsampling)
array_param, _, nn_score = make_st_nn(subkey,
                                      nn=my_nn, dim_in=d, batch_size=train_nsamples)

loss_type = args.loss_type
loss_fn = make_linear_sde_law_loss(sde, nn_score, t0=0., T=T, nsteps=train_nsteps,
                                   random_times=True, loss_type=loss_type, save_mem=args.save_mem)

nsteps_per_epoch = data_size // train_nsamples
if args.schedule == 'cos':
    until_steps = int(0.95 * nepochs) * nsteps_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=until_steps, alpha=1e-2)
elif args.schedule == 'exp':
    schedule = optax.exponential_decay(args.lr, data_size // train_nsamples, .96)
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

for i in range(nepochs):
    data_key, subkey = jax.random.split(data_key)
    perm_inds = dataset.init_enumeration(subkey, train_nsamples)
    for j in range(nsteps_per_epoch):
        subkey, subkey2 = jax.random.split(subkey)
        x0s = dataset.enumerate_subset(j, perm_inds, subkey)
        param, opt_state, loss = optax_kernel(param, opt_state, subkey2, x0s)
        ema_param = ema_kernel(ema_param, param, j, 300, 2, 0.99)
        print(f'{dataset_name} | {args.upsampling} | {args.sde} | {loss_type} | {args.schedule} | '
              f'Epoch: {i} / {nepochs}, iter: {j} / {data_size // train_nsamples}, loss: {loss:.4f}')
    if (i + 1) % 100 == 0:
        filename = f'./checkpoints/{dataset_name}_{args.sde}_{i}.npz'
        np.savez(filename, param=param, ema_param=ema_param)

print('Training done.')


# Verify if the score function is learnt properly
def rev_sim(key_, u0):
    def learnt_score(x, t):
        return nn_score(x, t, ema_param)

    return reverse_simulator(key_, u0, ts, learnt_score, sde.drift, sde.dispersion, integrator='euler-maruyama')


def sampler(key_):
    x, _, _ = dataset.sampler(key_)
    return x


# Simulate the backward and verify if it matches the target distribution
key = jax.random.PRNGKey(666)
key, subkey = jax.random.split(key)
test_sample0 = sampler(subkey)
key, subkey = jax.random.split(key)
traj = simulate_cond_forward(subkey, test_sample0, ts)
terminal_val = traj[-1]

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, None])(keys, terminal_val)
print(jnp.min(approx_init_samples), jnp.max(approx_init_samples))

fig, axes = plt.subplots(ncols=7, sharey='row')
axes[0].imshow(test_sample0)
axes[1].imshow(terminal_val)
for i in range(2, 7):
    axes[i].imshow(approx_init_samples[i - 2])
plt.tight_layout(pad=0.1)
plt.savefig(f'./tmp_figs/{dataset_name}_{args.sde}_backward_test.png')
plt.show()
