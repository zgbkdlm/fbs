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
from fbs.data import CelebAHQRestore, MNISTRestore
from fbs.data.images import normalise
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE
from fbs.samplers import gibbs_init, gibbs_kernel
from fbs.samplers.smc import pmcmc_kernel
from fbs.samplers.resampling import stratified
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='Training forward noising model.')
parser.add_argument('--dataset', type=str, default='mnist', help='Which dataset. Options are mnist, celeba-64, '
                                                                 'or celeba-128.')
parser.add_argument('--rect_size', type=int, default=15, help='The w/h of the inpainting rectangle.')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--method', type=str, default='gibbs', help='What method to do the conditional sampling. '
                                                                'Options are filter, gibbs, gibbs-eb, and pmcmc.')
parser.add_argument('--test_nsteps', type=int, default=500)
parser.add_argument('--test_epoch', type=int, default=2999)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--ny0s', type=int, default=10)
parser.add_argument('--nparticles', type=int, default=100)
parser.add_argument('--nsamples', type=int, default=10)
parser.add_argument('--init_method', type=str, default='smoother')
parser.add_argument('--marg', action='store_true', default=False, help='Whether marginalise our the Y path.')

args = parser.parse_args()
dataset_name = args.dataset
resolution = 28 if dataset_name == 'mnist' else int(dataset_name.split('-')[-1])
nchannels = 1 if dataset_name == 'mnist' else 3
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
    dataset = MNISTRestore(subkey, 'datasets/mnist.npz', task=f'inpaint-{rect_size}', test=True)
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
elif args.sde == 'exp':
    # Not tested.
    sde = StationaryExpLinearSDE(a=-0.5, b=1., c=1., z=1.)
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
x_shape = (rect_size ** 2, nchannels)
y_shape = (resolution ** 2 - rect_size ** 2, nchannels)


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


def transition_sampler(us_prev, v_prev, t_prev, key_, dataset_param):
    @partial(jax.vmap, in_axes=[0, None, None])
    def f(u, v, t):
        return reverse_drift_u(u, v, t, dataset_param)

    return (us_prev + f(us_prev, v_prev, t_prev) * dt
            + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))


def transition_logpdf(u, u_prev, v_prev, t_prev, dataset_param):
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def f(u, u_prev, v_prev, t_prev):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                                   u_prev + reverse_drift_u(u_prev, v_prev, t_prev, dataset_param) * dt,
                                                   math.sqrt(dt) * reverse_dispersion(t_prev)))

    return f(u, u_prev, v_prev, t_prev)


def likelihood_logpdf(v, u_prev, v_prev, t_prev, dataset_param):
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def f(v, u_prev, v_prev, t_prev):
        cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev, dataset_param) * dt
        return jnp.sum(jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * reverse_dispersion(t_prev)))

    return f(v, u_prev, v_prev, t_prev)


def fwd_sampler(key_, x0_, y0_, mask_):
    xy0 = dataset.concat(x0_, y0_, mask_)
    return simulate_cond_forward(key_, xy0, ts)


def fwd_ys_sampler(key_, y0_):
    return simulate_cond_forward(key_, y0_, ts)


def ref_sampler(key_, n):
    return jax.random.normal(key_, (n, *x_shape))


def ref_logpdf(x):
    return jnp.sum(jax.scipy.stats.norm.logpdf(x, 0., 1.))


pf = jax.jit(partial(gibbs_init, x0_shape=x_shape, ts=ts, fwd_sampler=fwd_sampler,
                     sde=sde, dataset=dataset,
                     transition_sampler=transition_sampler, transition_logpdf=transition_logpdf,
                     likelihood_logpdf=likelihood_logpdf,
                     nparticles=nparticles, method='filter', marg_y=args.marg))
debug = jax.jit(partial(gibbs_init, x0_shape=x_shape, ts=ts, fwd_sampler=fwd_sampler,
                        sde=sde, dataset=dataset,
                        transition_sampler=transition_sampler, transition_logpdf=transition_logpdf,
                        likelihood_logpdf=likelihood_logpdf,
                        nparticles=nparticles, method='debug', marg_y=args.marg))
gibbs_init = jax.jit(partial(gibbs_init, x0_shape=x_shape, ts=ts, fwd_sampler=fwd_sampler,
                             sde=sde, dataset=dataset,
                             transition_sampler=transition_sampler, transition_logpdf=transition_logpdf,
                             likelihood_logpdf=likelihood_logpdf,
                             nparticles=nparticles, method=args.init_method, marg_y=args.marg))
gibbs_kernel = jax.jit(partial(gibbs_kernel, ts=ts, fwd_sampler=fwd_sampler, sde=sde, dataset=dataset,
                               nparticles=nparticles, transition_sampler=transition_sampler,
                               transition_logpdf=transition_logpdf, likelihood_logpdf=likelihood_logpdf,
                               marg_y=args.marg, explicit_backward=True if args.method == 'gibbs-eb' else False))
pmcmc_kernel = jax.jit(partial(pmcmc_kernel, ts=ts, fwd_ys_sampler=fwd_ys_sampler, sde=sde,
                               ref_sampler=ref_sampler, ref_logpdf=ref_logpdf, transition_sampler=transition_sampler,
                               likelihood_logpdf=likelihood_logpdf, resampling=stratified,
                               nparticles=nparticles, delta=0.5))


def to_imsave(img):
    return img[..., 0] if nchannels == 1 else img


for k in range(args.ny0s):
    print(f'Running conditional sampler for {k}-th test sample.')
    key, subkey = jax.random.split(key)
    test_img, test_y0, mask = dataset.sampler(subkey)

    plt.imsave(f'./tmp_figs/{dataset_name}_inpainting-{rect_size}_{k}_true.png', to_imsave(test_img),
               cmap='gray' if nchannels == 1 else 'viridis')
    plt.imsave(f'./tmp_figs/{dataset_name}_inpainting-{rect_size}_{k}_corrupt.png',
               to_imsave(dataset.concat(jnp.zeros((rect_size ** 2, nchannels)), test_y0, mask)),
               cmap='gray' if nchannels == 1 else 'viridis')

    if args.method == 'filter':
        for i in range(nsamples):
            key, subkey = jax.random.split(key)
            x0, _ = pf(subkey, test_y0, dataset_param=mask)
            plt.imsave(f'./tmp_figs/{dataset_name}_inpainting-{rect_size}'
                       f'_filter{"_marg" if args.marg else ""}_{k}_{i}.png',
                       to_imsave(dataset.concat(x0, test_y0, mask)),
                       cmap='gray' if nchannels == 1 else 'viridis')
            print(f'Inpainting-{rect_size} | filter | iter: {i}')
    elif args.method == 'debug':
        key, subkey = jax.random.split(key)
        x0s, _ = debug(subkey, test_y0, dataset_param=mask)
        np.save(f'x0s-filter-{k}', x0s)
    elif 'gibbs' in args.method:
        key, subkey = jax.random.split(key)
        x0, us_star = gibbs_init(subkey, test_y0, dataset_param=mask)
        bs_star = jnp.zeros((nsteps + 1), dtype=int)
        plt.imsave(f'./tmp_figs/{dataset_name}_inpainting-{rect_size}_gibbs_{k}_init.png',
                   to_imsave(dataset.concat(x0, test_y0, mask)),
                   cmap='gray' if nchannels == 1 else 'viridis')

        for i in range(nsamples):
            key, subkey = jax.random.split(key)
            x0, us_star, bs_star, acc = gibbs_kernel(subkey, x0, test_y0, us_star, bs_star, dataset_param=mask)
            sample = us_star[-1]
            plt.imsave(
                f'./tmp_figs/{dataset_name}_inpainting-{rect_size}_gibbs{"_marg" if args.marg else ""}_{k}_{i}.png',
                to_imsave(dataset.concat(us_star[-1], test_y0, mask)),
                cmap='gray' if nchannels == 1 else 'viridis')

            print(f'Inpainting-{rect_size} | Gibbs | iter: {i}, acc: {acc}')
    elif args.method == 'pmcmc':
        x0, log_ell, yT, xT = jnp.zeros(x_shape), 0., jnp.zeros(y_shape), jnp.zeros(x_shape)
        for i in range(nsamples):
            key, subkey = jax.random.split(key)
            x0, log_ell, yT, xT, mcmc_state = pmcmc_kernel(subkey, x0, log_ell, yT, xT, test_y0, dataset_param=mask)
            plt.imsave(
                f'./tmp_figs/{dataset_name}_inpainting-{rect_size}_pmcmc_{k}_{i}.png',
                to_imsave(dataset.concat(x0, test_y0, mask)), cmap='gray' if nchannels == 1 else 'viridis')
            print(f'Inpainting-{rect_size} | pMCMC | iter: {i}, acc_prob: {mcmc_state.acceptance_prob}')
    else:
        raise ValueError(f"Unknown method {args.method}")
