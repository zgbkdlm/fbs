r"""
Training a Schrodinger bridge.

The reference terminal distribution is N(0, 1).

Run the script under the folder `./experiments`.
"""
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fbs.data import CelebAHQRestore, MNISTRestore
from fbs.sdes import StationaryConstLinearSDE, StationaryLinLinearSDE, euler_maruyama
from fbs.dsb.base import ipf_loss_cont, ipf_loss_cont_v
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet

# Parse arguments
parser = argparse.ArgumentParser(description='Training a Schrodinger bridge for images.')
parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset. Options are mnist, celeba-64, '
                                                                 'or celeba-128.')
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument('--sde', type=str, default='lin', help='The reference SDE.')
parser.add_argument('--vmap_loss', action='store_true', default=False)
parser.add_argument('--upsampling', type=str, default='pixel_shuffle')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--nn_dim', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--nsteps', type=int, default=2)
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--nsbs', type=int, default=10, help='The number of Schrodinger iterations.')
parser.add_argument('--grad_clip', action='store_true', default=False)

args = parser.parse_args()
dataset_name = args.dataset
resolution = 28 if dataset_name == 'mnist' else int(dataset_name.split('-')[-1])

print(f'Train Schrodinger bridge on {args.dataset}')

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, key_sb = jax.random.split(key)

T = args.T
nsteps = 128
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

# Define the reference SDE
if args.sde == 'const':
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
elif args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=5., t0=0., T=T)
else:
    raise NotImplementedError('...')


def reference_drift(x, t, _):
    return sde.drift(x, t)


# NN
train_nsamples = args.batch_size
train_nsteps = args.nsteps
nn_dt = 0.5 / 200
nepochs = args.nepochs
data_size = dataset.n

key, subkey = jax.random.split(key)
my_nn = UNet(dt=nn_dt, dim=args.nn_dim, upsampling=args.upsampling)
param_fwd, _, nn_drift = make_st_nn(subkey,
                                    nn=my_nn, dim_in=d, batch_size=train_nsamples)
param_bwd, _, _ = make_st_nn(subkey,
                             nn=my_nn, dim_in=d, batch_size=train_nsamples)

# Set up Optax
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

# Make loss functions
ipf_loss = ipf_loss_cont_v if args.vmap_loss else ipf_loss_cont


def loss_fn_init(param_bwd_, param_fwd_, key_, data_samples):
    """Simulate the forward data -> sth. to learn its backward.
    This loss is for the first iteration that uses the reference SDE.
    """
    key_loss, key_ts = jax.random.split(key_, num=2)
    rnd_ts = jnp.hstack([0.,
                         jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0. + 1e-5, maxval=T)),
                         T])
    return ipf_loss(key_loss, param_bwd_, param_fwd_, data_samples, rnd_ts, nn_drift,
                    reference_drift, sde.dispersion)


def loss_fn_bwd(param_bwd_, param_fwd_, key_, data_samples):
    """Simulate the forward data -> sth. to learn its backward.
    """
    key_loss, key_ts = jax.random.split(key_, num=2)
    rnd_ts = jnp.hstack([0.,
                         jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0. + 1e-5, maxval=T)),
                         T])
    return ipf_loss(key_loss, param_bwd_, param_fwd_, data_samples, rnd_ts, nn_drift, nn_drift, sde.dispersion)


def loss_fn_fwd(param_fwd_, param_bwd_, key_, ref_samples):
    """Simulate the backward sth. <- ref to learn its forward.
    """
    key_loss, key_ts = jax.random.split(key_, num=2)
    rnd_ts = jnp.hstack([0.,
                         jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0. + 1e-5, maxval=T)),
                         T])
    return ipf_loss(key_loss, param_fwd_, param_bwd_, ref_samples, T - rnd_ts, nn_drift, nn_drift, sde.dispersion)


# Make optax kernels
@jax.jit
def optax_kernel_init(param_bwd_, opt_state_, param_fwd_, key_, data_samples):
    loss, grad = jax.value_and_grad(loss_fn_init)(param_bwd_, param_fwd_, key_, data_samples)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_bwd_)
    param_bwd_ = optax.apply_updates(param_bwd_, updates)
    return param_bwd_, opt_state_, loss


@jax.jit
def optax_kernel_bwd(param_bwd_, opt_state_, param_fwd_, key_, data_samples):
    loss, grad = jax.value_and_grad(loss_fn_bwd)(param_bwd_, param_fwd_, key_, data_samples)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_bwd_)
    param_bwd_ = optax.apply_updates(param_bwd_, updates)
    return param_bwd_, opt_state_, loss


@jax.jit
def optax_kernel_fwd(param_fwd_, opt_state_, param_bwd_, key_, ref_samples):
    loss, grad = jax.value_and_grad(loss_fn_fwd)(param_fwd_, param_bwd_, key_, ref_samples)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_fwd_)
    param_fwd_ = optax.apply_updates(param_fwd_, updates)
    return param_fwd_, opt_state_, loss


# Make a loop for Schrodinger bridge iteration
def sb_kernel(param_fwd_, param_bwd_, opt_state_fwd_, opt_state_bwd_, key_data, sb_step: int):
    # Learn the backward process, i.e., data <- ref
    for i in range(nepochs):
        key_data, subkey = jax.random.split(key_data)
        perm_inds = dataset.init_enumeration(subkey, train_nsamples)
        for j in range(nsteps_per_epoch):
            subkey, subkey2 = jax.random.split(subkey)
            x0s = dataset.enumerate_subset(j, perm_inds, subkey)
            if sb_step == 0:
                param_bwd_, opt_state_bwd_, loss = optax_kernel_init(param_bwd_, opt_state_bwd_, param_fwd_, subkey2,
                                                                     x0s)
            else:
                param_bwd_, opt_state_bwd_, loss = optax_kernel_bwd(param_bwd_, opt_state_bwd_, param_fwd_, subkey2,
                                                                    x0s)
            print(f'{dataset_name} | Learning bwd ({args.sde}) | SB iter: {sb_step} | Epoch: {i} / {nepochs}, '
                  f'iter: {j} / {data_size // train_nsamples} | loss: {loss:.4f}')

    # Learn the forward process, i.e., data -> ref
    for i in range(nepochs):
        key_data, subkey = jax.random.split(key_data)
        for j in range(nsteps_per_epoch):
            subkey, subkey2 = jax.random.split(subkey)
            xTs = jax.random.normal(subkey, (train_nsamples, *d))
            param_fwd_, opt_state_fwd_, loss = optax_kernel_fwd(param_fwd_, opt_state_fwd_, param_bwd_, subkey2, xTs)
            print(f'{dataset_name} | Learning fwd ({args.sde}) | SB iter: {sb_step} | Epoch: {i} / {nepochs}, '
                  f'iter: {j} / {data_size // train_nsamples} | loss: {loss:.4f}')

    return param_fwd_, param_bwd_, opt_state_fwd_, opt_state_bwd_


# Now the Scrodinger bridge loop
opt_state_fwd = optimiser.init(param_fwd)
opt_state_bwd = optimiser.init(param_bwd)

for sb_iter in range(args.nsbs):
    key_sb, subkey = jax.random.split(key_sb)
    param_fwd, param_bwd, opt_state_fwd, opt_state_bwd = sb_kernel(param_fwd, param_bwd, opt_state_fwd, opt_state_bwd,
                                                                   subkey, sb_iter)

    filename = f'./checkpoints/sb_{dataset_name}_{args.sde}_{sb_iter}.npz'
    np.savez(filename, param_fwd=param_fwd, param_bwd=param_bwd)


# Verify if the Schrodinger bridge is learnt properly
def rev_sim(key_, xT):
    def rev_drift(x, t):
        return nn_drift(x, t, param_bwd)

    return euler_maruyama(key_, xT, T - ts, rev_drift, sde.dispersion, integration_nsteps=10, return_path=False)


def fwd_sim(key_, x0):
    def fwd_drift(x, t):
        return nn_drift(x, t, param_fwd)

    return euler_maruyama(key_, x0, ts, fwd_drift, sde.dispersion, integration_nsteps=10, return_path=False)


key = jax.random.PRNGKey(666)
key, subkey = jax.random.split(key)
ref_samples = jax.random.normal(subkey, (5, *d))

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, 0])(keys, ref_samples)
print(jnp.min(approx_init_samples), jnp.max(approx_init_samples))

fig, axes = plt.subplots(ncols=5, sharey='row')
for i in range(5):
    axes[i].imshow(approx_init_samples[i])
plt.tight_layout(pad=0.1)
plt.savefig(f'./tmp_figs/sb_{dataset_name}_{args.sde}_test.png')
plt.show()
