r"""
Inpainting experiment using the twisted diffusion sampler.

Run the script under the folder `./experiments`.
"""
import math
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fbs.data import CelebAHQRestore, MNISTRestore
from fbs.data.images import normalise
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, \
    StationaryLinLinearSDE
from fbs.samplers import twisted_smc, stratified
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='Inpainting using the twisted diffusion sampler.')
parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset. Options are mnist, celeba-64, '
                                                                 'or celeba-128.')
parser.add_argument('--rect_size', type=int, default=15, help='The w/h of the inpainting rectangle.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--test_nsteps', type=int, default=500)
parser.add_argument('--test_epoch', type=int, default=2999)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--ny0s', type=int, default=10)
parser.add_argument('--start_from', type=int, default=0, help='Star from which y0.')
parser.add_argument('--nparticles', type=int, default=100)
parser.add_argument('--nsamples', type=int, default=10)

args = parser.parse_args()
dataset_name = args.dataset
resolution = 28 if dataset_name == 'mnist' else int(dataset_name.split('-')[-1])
nchannels = 1 if dataset_name == 'mnist' else 3
cmap = 'gray' if nchannels == 1 else 'viridis'
rect_size = args.rect_size

print(f'Test inpainting-{rect_size} on {args.dataset}')

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
nsamples = args.nsamples
data_variance = 0.06  # According to the original paper, the variance of the dataset.
x_shape = (rect_size ** 2, nchannels)
y_shape = (resolution ** 2 - rect_size ** 2, nchannels)
xy_shape = (resolution, resolution, nchannels)


def unpack(xy, mask_):
    return dataset.unpack(xy, mask_)


def reverse_drift(uv, t):
    return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * nn_score(uv, T - t, param)


def reverse_cond_drift(uv, t, y, mask_):
    return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * (
            nn_score(uv, T - t, param) + jax.grad(twisting_logpdf, argnums=1)(y, uv, t, mask_))


def reverse_dispersion(t):
    return sde.dispersion(T - t)


@partial(jax.vmap, in_axes=[0, 0, None])
def transition_logpdf(u, u_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                               u_prev + reverse_drift(u_prev, t_prev) * dt,
                                               math.sqrt(dt) * reverse_dispersion(t_prev)))


def init_sampler(key_, nparticles_):
    return jax.random.normal(key_, (nparticles_, *xy_shape))


def twisting_logpdf(y, uv, t, mask_):
    denoising_estimate = uv + reverse_drift(uv, t) * dt
    _, obs_part = unpack(denoising_estimate, mask_)
    F, Q = discretise_linear_sde(T - t, ts[0])
    return jnp.sum(jax.scipy.stats.norm.logpdf(y, obs_part, jnp.sqrt(F ** 2 * data_variance + Q)))


def twisting_logpdf_vmap(y, uvs, t, mask_):
    f = lambda q, w, e: twisting_logpdf(q, w, e, mask_)
    return jax.vmap(f, in_axes=[None, 0, None])(y, uvs, t)


def twisting_prop_sampler(key_, uvs, t, y, mask_):
    _reverse_cond_drift = lambda q, w, e: reverse_cond_drift(q, w, e, mask_)
    m_ = uvs + jax.vmap(_reverse_cond_drift, in_axes=[0, None, None])(uvs, t, y) * dt
    return m_ + math.sqrt(dt) * reverse_dispersion(t) * jax.random.normal(key_, (nparticles, *xy_shape))


def twisting_prop_logpdf(u, u_prev, t, y, mask_):
    def f(u, u_prev, t, y):
        m_ = u_prev + reverse_cond_drift(u_prev, t, y, mask_) * dt
        return jnp.sum(jax.scipy.stats.norm.logpdf(u, m_, math.sqrt(dt) * reverse_dispersion(t)))

    return jax.vmap(f, in_axes=[0, 0, None, None])(u, u_prev, t, y)


@jax.jit
def conditional_sampler(key_, y, **kwargs):
    key_filter, key_select = jax.random.split(key_)
    uvs, log_ws = twisted_smc(key_filter, y, ts,
                              init_sampler, transition_logpdf, twisting_logpdf_vmap, twisting_prop_sampler,
                              twisting_prop_logpdf,
                              resampling=stratified, nparticles=nparticles, **kwargs)
    return jax.random.choice(key_select, uvs, p=jnp.exp(log_ws), axis=0)


def to_imsave(img):
    img = normalise(img, method='clip')
    return img[..., 0] if nchannels == 1 else img


@jax.jit
def dataset_sampler(key_):
    return dataset.sampler(key_)


for k in range(args.ny0s):
    print(f'Running conditional sampler for {k}-th test sample.')
    data_key, subkey = jax.random.split(data_key)
    if k < args.start_from:
        continue
    test_img, test_y0, mask = dataset_sampler(subkey)
    path_head_img = f'./imgs/results_inpainting/imgs/{dataset_name}-{rect_size}-{args.sde}-{nparticles}-{k}'
    path_head_arr = f'./imgs/results_inpainting/arrs/{dataset_name}-{rect_size}-{args.sde}-{nparticles}-{k}'

    plt.imsave(path_head_img + '-true.png', to_imsave(test_img), cmap=cmap)
    np.savez(path_head_arr + '-true', test_img=test_img, *mask)
    plt.imsave(path_head_img + '-corrupt.png',
               to_imsave(dataset.concat(jnp.zeros(x_shape), test_y0, mask)),
               cmap=cmap)

    restored_imgs = np.zeros((nsamples, resolution, resolution, nchannels))

    for i in range(nsamples):
        key, subkey = jax.random.split(key)
        x0 = conditional_sampler(subkey, test_y0, mask_=mask)
        restored_imgs[i] = x0
        plt.imsave(path_head_img + f'-twisted-{i}.png', to_imsave(x0), cmap=cmap)
        print(f'Inpainting-{rect_size} | Twisted | iter: {i}')
    np.save(path_head_arr + f'-twisted', restored_imgs)
