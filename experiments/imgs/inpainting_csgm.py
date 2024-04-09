r"""
Inpainting experiment using the standard conditional score matching.

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
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet

# Parse arguments
parser = argparse.ArgumentParser(description='Inpainting using the standard conditional score matching.')
parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset. Options are mnist, celeba-64, '
                                                                 'or celeba-128.')
parser.add_argument('--rect_size', type=int, default=15, help='The w/h of the inpainting rectangle.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--test_nsteps', type=int, default=500)
parser.add_argument('--test_epoch', type=int, default=2999)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--ny0s', type=int, default=10)
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
nsamples = args.nsamples
x_shape = (rect_size ** 2, nchannels)
y_shape = (resolution ** 2 - rect_size ** 2, nchannels)


def reverse_drift(u, t, mask_, key_, y0):
    F, Q = discretise_linear_sde(T - t, ts[0])
    v_hat = F * y0 + jnp.sqrt(Q) * jax.random.normal(key_, y_shape)
    uv = dataset.concat(u, v_hat, mask_)
    return -sde.drift(u, T - t) + sde.dispersion(T - t) ** 2 * dataset.unpack(nn_score(uv, T - t, param), mask_)[0]


def reverse_dispersion(t):
    return sde.dispersion(T - t)


def cond_ref_sampler(key_, y):
    return jax.random.normal(key_, x_shape)


def euler_maruyama(key_, u0, mask_, y0):
    def scan_body(carry, elem):
        u = carry
        rnd, t, key_drift = elem

        u = u + reverse_drift(u, t, mask_, key_drift, y0) * dt + reverse_dispersion(t) * math.sqrt(dt) * rnd
        return u, None

    key_scan, key_est = jax.random.split(key_)
    key_ests = jax.random.split(key_est, num=nsteps)
    rnds = jax.random.normal(key_scan, (nsteps, *x_shape))
    return jax.lax.scan(scan_body, u0, (rnds, ts[:-1], key_ests))[0]


@jax.jit
def conditional_sampler(key_, y, mask_):
    key_init, key_sde = jax.random.split(key_, num=2)
    u0 = cond_ref_sampler(key_init, y)
    uT = euler_maruyama(key_sde, u0, mask_, y)
    return uT


def to_imsave(img):
    img = normalise(img, method='clip')
    return img[..., 0] if nchannels == 1 else img


@jax.jit
def dataset_sampler(key_):
    return dataset.sampler(key_)


for k in range(args.ny0s):
    print(f'Running conditional sampler for {k}-th test sample.')
    key, subkey = jax.random.split(key)
    test_img, test_y0, mask = dataset_sampler(subkey)
    path_head_img = f'./imgs/results_inpainting/imgs/{dataset_name}-{rect_size}-{k}'
    path_head_arr = f'./imgs/results_inpainting/arrs/{dataset_name}-{rect_size}-{k}'

    plt.imsave(path_head_img + '-true.png', to_imsave(test_img), cmap=cmap)
    np.save(path_head_arr + '-true', test_img)
    plt.imsave(path_head_img + '-corrupt.png',
               to_imsave(dataset.concat(jnp.zeros(x_shape), test_y0, mask)),
               cmap=cmap)

    for i in range(nsamples):
        key, subkey = jax.random.split(key)
        x0 = conditional_sampler(subkey, test_y0, mask)
        plt.imsave(path_head_img + f'-csgm-{i}.png', to_imsave(x0), cmap=cmap)
        np.save(path_head_arr + f'-csgm-{i}', x0)
        print(f'Inpainting-{rect_size} | cSGM | iter: {i}')
