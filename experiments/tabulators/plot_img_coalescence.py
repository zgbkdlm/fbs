r"""
Demonstrate the coalescence effect of the filtering method.

Run the script under the folder `./experiments`.
"""
import math
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fbs.data import CelebAHQRestore, MNISTRestore
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE
from fbs.samplers import gibbs_init
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='Coalescence.')
parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset. Options are mnist, celeba-64, '
                                                                 'or celeba-128.')
parser.add_argument('--rect_size', type=int, default=15, help='The w/h of the inpainting rectangle.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--test_nsteps', type=int, default=1000)
parser.add_argument('--test_epoch', type=int, default=2999)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--nparticles', type=int, default=10)

args = parser.parse_args()
dataset_name = args.dataset
resolution = 28 if dataset_name == 'mnist' else int(dataset_name.split('-')[-1])
nchannels = 1 if dataset_name == 'mnist' else 3
cmap = 'gray' if nchannels == 1 else 'viridis'
rect_size = args.rect_size

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(args.test_seed)
key, data_key = jax.random.split(key)

T = 2
nsteps = args.test_nsteps
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# Load dataset
key, subkey = jax.random.split(key)
if dataset_name == 'mnist':
    d = (resolution, resolution, 1)
    dataset = MNISTRestore(subkey, '../datasets/mnist.npz', task=f'inpaint-{rect_size}', test=True)
elif 'celeba' in dataset_name:
    d = (resolution, resolution, 3)
    dataset = CelebAHQRestore(subkey, f'datasets/celeba_hq{resolution}.npy',
                              task=f'inpaint-{rect_size}', resolution=resolution, test=True)
else:
    raise NotImplementedError(f'{dataset_name} not implemented.')

# Define the forward noising process
if args.sde == 'const':
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
elif args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=5., t0=0., T=T)
else:
    raise NotImplementedError('...')
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

# Load the trained model
key, subkey = jax.random.split(key)
my_nn = UNet(dt=T / 200, dim=64, upsampling='pixel_shuffle')
_, _, nn_score = make_st_nn(subkey, nn=my_nn, dim_in=d, batch_size=2)

filename = f'./checkpoints/{dataset_name}_{args.sde}_{args.test_epoch}.npz'
param = np.load(filename)['ema_param' if args.test_ema else 'param']

# Conditional sampling
nparticles = args.nparticles
x_shape = (rect_size ** 2, nchannels)
y_shape = (resolution ** 2 - rect_size ** 2, nchannels)


def unpack(xy, mask_):
    return dataset.unpack(xy, mask_)


def reverse_drift(uv, t):
    return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * nn_score(uv, T - t, param)


def reverse_drift_u(u, v, t, mask_):
    drift = reverse_drift(dataset.concat(u, v, mask_), t)
    rdu, rdv = dataset.unpack(drift, mask_)
    return rdu


def reverse_drift_v(v, u, t, mask_):
    drift = reverse_drift(dataset.concat(u, v, mask_), t)
    rdu, rdv = dataset.unpack(drift, mask_)
    return rdv


def reverse_dispersion(t):
    return sde.dispersion(T - t)


def transition_sampler(us_prev, v_prev, t_prev, key_, mask_):
    @partial(jax.vmap, in_axes=[0, None, None])
    def f(u, v, t):
        return reverse_drift_u(u, v, t, mask_)

    return (us_prev + f(us_prev, v_prev, t_prev) * dt
            + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))


def transition_logpdf(u, u_prev, v_prev, t_prev, mask_):
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def f(u, u_prev, v_prev, t_prev):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                                   u_prev + reverse_drift_u(u_prev, v_prev, t_prev, mask_) * dt,
                                                   math.sqrt(dt) * reverse_dispersion(t_prev)))

    return f(u, u_prev, v_prev, t_prev)


def likelihood_logpdf(v, u_prev, v_prev, t_prev, mask_):
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def f(v, u_prev, v_prev, t_prev):
        cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev, mask_) * dt
        return jnp.sum(jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * reverse_dispersion(t_prev)))

    return f(v, u_prev, v_prev, t_prev)


def fwd_sampler(key_, x0_, y0_, mask_):
    xy0 = dataset.concat(x0_, y0_, mask_)
    return simulate_cond_forward(key_, xy0, ts)


def ref_sampler(key_, _, n):
    return jax.random.normal(key_, (n, *x_shape))


pf = jax.jit(partial(gibbs_init, x0_shape=x_shape, ts=ts, fwd_sampler=fwd_sampler,
                     sde=sde, unpack=unpack,
                     transition_sampler=transition_sampler, transition_logpdf=transition_logpdf,
                     likelihood_logpdf=likelihood_logpdf,
                     nparticles=nparticles, method='debug', marg_y=False))


@jax.jit
def dataset_sampler(key_):
    return dataset.sampler(key_)


# Run
data_key, subkey = jax.random.split(data_key)
test_img, test_y0, mask = dataset_sampler(subkey)

key, subkey = jax.random.split(key)
x0s, _ = pf(subkey, test_y0, mask_=mask)
x0s = np.reshape(x0s, (nsteps + 1, nparticles, -1))
which_d = 6

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Plot the coalescence of one dimension
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
every = nsteps // 10  # Reduce image size

for p in range(nparticles):
    if p == 0:
        line1, = axes[0].plot(ts[::every], x0s[::every, p, which_d], linewidth=1, c='black', alpha=0.3,
                              label='Particle trajectory')
    else:
        axes[0].plot(ts[::every], x0s[::every, p, which_d], linewidth=1, c='black', alpha=0.3)

axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('Particle value')
axes[0].legend(handles=[line1])

# Plot the variances of the particles for all dimensions
two_sigmas = 2 * np.sqrt(np.var(x0s, axis=1))
quantile = np.quantile(two_sigmas, 0.95, axis=-1)
for d in range(x0s.shape[-1]):
    if d == 0:
        line2, = axes[1].plot(ts[::every], two_sigmas[::every, d], linewidth=1, c='black', alpha=0.1,
                              label=r'Particles $2\,\sigma$ for each dimension')
    else:
        axes[1].plot(ts[::every], two_sigmas[::every, d], linewidth=1, c='black', alpha=0.1)
ql, = axes[1].plot(ts[::every], quantile[::every], linewidth=3, c='black', label='0.95 quantile over all dimensions')
# axes[1].annotate(f'{quantile[-1]:.2f}', xy=(ts[-1], quantile[-1]), xytext=(ts[-1] - 0.3, quantile[-1] + 0.8),
#                  arrowprops=dict(facecolor='black', width=2, shrink=0.05, alpha=0.5))

axes[1].grid(linestyle='--', alpha=0.3, which='both')
axes[1].set_xlabel('$t$')
axes[1].set_ylabel(r'Particles $2\,\sigma$')
axes[1].legend(handles=[line2, ql])

plt.tight_layout(pad=0.1)
plt.legend()
plt.savefig(f'./figs/coalescence-{args.sde}-{args.nparticles}.pdf', transparent=True)
plt.show()
