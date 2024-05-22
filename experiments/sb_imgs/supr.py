r"""
MNIST super-resolution under a Schrodinger bridge model.

The aim is to show the effect of x0.
"""
import math
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fbs.data import MNISTRestore
from fbs.data.images import normalise
from fbs.sdes import StationaryConstLinearSDE, StationaryLinLinearSDE
from fbs.sdes.simulators import euler_maruyama
from fbs.samplers import gibbs_kernel as _gibbs_kernel, gibbs_init as _gibbs_init
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='Super-resolution.')
parser.add_argument('--rate', type=int, default=4, help='The rate of super-resolution.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--method', type=str, default='filter')
parser.add_argument('--test_nsteps', type=int, default=128)
parser.add_argument('--sb_step', type=int, default=9)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--y0_id', type=int, default=10)
parser.add_argument('--nparticles', type=int, default=100)
parser.add_argument('--nsamples', type=int, default=100)
parser.add_argument('--init_method', type=str, default='smoother')

args = parser.parse_args()
dataset_name = 'mnist'
resolution = 28
nchannels = 1
cmap = 'gray' if nchannels == 1 else 'viridis'
sr_rate = args.rate

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(args.test_seed)
key, data_key = jax.random.split(key)

T = 0.5
nsteps = args.test_nsteps
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# Load dataset
key, subkey = jax.random.split(key)
d = (resolution, resolution, 1)
dataset = MNISTRestore(subkey, '../datasets/mnist.npz', task=f'supr-{sr_rate}', test=True)
dataset.sr_random = False

# Define the forward noising process
if args.sde == 'const':
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
elif args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=5., t0=0., T=T)
else:
    raise NotImplementedError('...')

# Load the trained model
key, subkey = jax.random.split(key)
my_nn = UNet(dt=0.5 / 200, dim=64, upsampling='pixel_shuffle')
_, _, nn_drift = make_st_nn(subkey, nn=my_nn, dim_in=d, batch_size=2)

filename = f'./checkpoints/sb_{dataset_name}_{args.sde}_{args.sb_step}.npz'
param_fwd, param_bwd = np.load(filename)['param_fwd'], np.load(filename)['param_bwd']

# Conditional sampling
nparticles = args.nparticles
nsamples = args.nsamples
x_shape = dataset.unobs_shape
y_shape = (resolution ** 2 - x_shape[0], nchannels)


def unpack(xy, mask_):
    return dataset.unpack(xy, mask_)


def reverse_drift(uv, t):
    return nn_drift(uv, T - t, param_bwd)


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
    def fwd_drift(x, t):
        return nn_drift(x, t, param_fwd)

    xy0 = dataset.concat(x0_, y0_, mask_)
    return euler_maruyama(key_, xy0, ts, fwd_drift, sde.dispersion, integration_nsteps=1, return_path=True)


def ref_sampler(key_, _, n):
    return jax.random.normal(key_, (n, *x_shape))


@jax.jit
def random_x0_sampler(key_, y0_, mask_):
    return jax.random.uniform(key_, x_shape, minval=0., maxval=1.)


@jax.jit
def blank_x0_sampler(key_, y0_, mask_):
    return jnp.zeros(x_shape)


@jax.jit
def interp_x0_sampler(key_, y0_, mask_):
    interpolated_img = jax.image.resize(jnp.reshape(y0_, (low_res, low_res, nchannels)),
                                        (resolution, resolution, nchannels),
                                        method='linear')
    return unpack(interpolated_img, mask_)[0]


@jax.jit
def pf(key_, x0_, y0_, mask_):
    return _gibbs_init(key_, y0_, x_shape, ts, fwd_sampler, sde, unpack,
                       transition_sampler, transition_logpdf, likelihood_logpdf,
                       nparticles, method='filter', marg_y=False, x0=x0_, mask_=mask_)


@jax.jit
def gibbs_init(key_, x0_, y0_, mask_):
    return _gibbs_init(key_, y0_, x_shape, ts, fwd_sampler, sde, unpack,
                       transition_sampler, transition_logpdf, likelihood_logpdf,
                       nparticles, method='smoother', marg_y=False, x0=x0_, mask_=mask_)


@jax.jit
def gibbs_kernel(key_, x0_, y0_, us_star_, bs_star_, mask_):
    return _gibbs_kernel(key_, x0_, y0_, us_star_, bs_star_,
                         ts, fwd_sampler, sde, unpack, nparticles,
                         transition_sampler, transition_logpdf, likelihood_logpdf,
                         marg_y=False, explicit_backward=True, explicit_final=True, mask_=mask_)


def to_imsave(img):
    img = normalise(img, method='clip')
    return img[..., 0] if nchannels == 1 else img


@jax.jit
def dataset_sampler(key_):
    return dataset.sampler(key_)


# Do
data_key, subkey = jax.random.split(data_key)
for _ in range(args.y0_id):
    data_key, subkey = jax.random.split(data_key)

test_img, test_y0, mask = dataset_sampler(subkey)
path_head_img = f'./sb_imgs/results/{dataset_name}-{sr_rate}-{args.sde}-{nparticles}-{args.y0_id}'
path_head_arr = f'./sb_imgs/results/{dataset_name}-{sr_rate}-{args.sde}-{nparticles}-{args.y0_id}'

plt.imsave(path_head_img + '-true.png', to_imsave(test_img), cmap=cmap)
np.savez(path_head_arr + '-true', test_img=test_img, *mask)
plt.imsave(path_head_img + '-corrupt.png',
           to_imsave(dataset.concat(jnp.zeros(x_shape), test_y0, mask)),
           cmap=cmap)
low_res = resolution // sr_rate
plt.imsave(path_head_img + '-corrupt-lr.png',
           to_imsave(jnp.reshape(test_y0, (low_res, low_res, nchannels))),
           cmap=cmap)

restored_imgs = np.zeros((nsamples, resolution, resolution, nchannels))

# Do conditional sampling
for x0_sampler, x0_sampler_name in zip([random_x0_sampler, blank_x0_sampler, interp_x0_sampler],
                                       ['random', 'blank', 'interp']):
    if args.method == 'filter':
        for i in range(nsamples):
            key, subkey = jax.random.split(key)
            x0 = x0_sampler(subkey, test_y0, mask_=mask)
            key, subkey = jax.random.split(key)
            x0, _ = pf(subkey, x0, test_y0, mask)
            restored = dataset.concat(x0, test_y0, mask)
            restored_imgs[i] = restored
            plt.imsave(path_head_img + f'-filter-{x0_sampler_name}-{i}.png', to_imsave(restored), cmap=cmap)
            print(f'Supr-{sr_rate} | Filter | {x0_sampler_name} | iter: {i}')
        np.save(path_head_arr + f'-filter-{x0_sampler_name}', restored_imgs)
    elif 'gibbs' in args.method:
        key, subkey = jax.random.split(key)
        x0 = x0_sampler(subkey, test_y0, mask_=mask)
        key, subkey = jax.random.split(key)
        x0, us_star = gibbs_init(subkey, x0, test_y0, mask)
        bs_star = jnp.zeros((nsteps + 1), dtype=int)
        restored = dataset.concat(x0, test_y0, mask)
        plt.imsave(path_head_img + '-gibbs-init.png', to_imsave(restored), cmap=cmap)
        np.save(path_head_arr + '-gibbs-init', restored)

        for i in range(nsamples):
            key, subkey = jax.random.split(key)
            x0, us_star, bs_star, acc = gibbs_kernel(subkey, x0, test_y0, us_star, bs_star, mask)
            restored = dataset.concat(x0, test_y0, mask)
            restored_imgs[i] = restored
            plt.imsave(
                path_head_img + f'-gibbs-eb-ef-{x0_sampler_name}-{i}.png',
                to_imsave(restored),
                cmap=cmap)
            print(f'Inpainting-{sr_rate} | Gibbs | {x0_sampler_name} | iter: {i}, acc: {acc}')
        np.save(
            path_head_arr + f'-gibbs-eb-ef-{x0_sampler_name}',
            restored_imgs)
    else:
        raise ValueError(f"Unknown method {args.method}")
