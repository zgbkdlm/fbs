r"""
Conditional sampling on CelebA HQ dataset.
"""
import math
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fbs.data import CelebAHQ
from fbs.data.images import normalise_rgb
from fbs.sdes import make_linear_sde, make_linear_sde_law_loss, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE, reverse_simulator
from fbs.sdes.simulators import doob_bridge_simulator
from fbs.filters.csmc.csmc import csmc_kernel
from fbs.filters.csmc.resamplings import killing
from fbs.nn.models import make_st_nn
from fbs.nn.unet_cifar import UNet
from fbs.nn.utils import make_optax_kernel
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='CelebA test.')
parser.add_argument('--train', action='store_true', default=False, help='Whether train or not.')
parser.add_argument('--task', type=str, default='supr')
parser.add_argument('--resolution', type=int, default=64)
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
parser.add_argument('--test_nsteps', type=int, default=500)
parser.add_argument('--test_epoch', type=int, default=39)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--nparticles', type=int, default=100)
parser.add_argument('--ngibbs', type=int, default=10)
parser.add_argument('--doob', action='store_true', default=False)

args = parser.parse_args()
train = args.train
task = args.task
resolution = args.resolution

print(f'{"Train" if train else "Test"} {task} on CelebaHQ{resolution}')

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 2
nsteps = args.test_nsteps
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# CIFAR10
d = (resolution, resolution, 6)
key, subkey = jax.random.split(key)
dataset = CelebAHQ(subkey, f'../datasets/celeba_hq{resolution}.npy', task=task, resolution=resolution)
dataset_test = CelebAHQ(subkey, f'../datasets/celeba_hq{resolution}.npy', task=task, resolution=resolution, test=True)


def sampler(key_, test: bool = False):
    x_, y_ = dataset_test.sampler(key_) if test else dataset.sampler(key_)
    return dataset.concat(x_, y_)


key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 4)
xys = jax.vmap(sampler, in_axes=[0])(keys)

if not train:
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for row in range(2):
        for col in range(4):
            axes[row, col].imshow(normalise_rgb(xys[col, :, :, row * 3:(row + 1) * 3]))
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

if train:
    for i in range(nepochs):
        data_key, subkey = jax.random.split(data_key)
        perm_inds = dataset.init_enumeration(subkey, train_nsamples)
        for j in range(nsteps_per_epoch):
            subkey, subkey2 = jax.random.split(subkey)
            x0s, y0s = dataset.enumerate_subset(j, perm_inds, subkey)
            xy0s = dataset.concat(x0s, y0s)
            param, opt_state, loss = optax_kernel(param, opt_state, subkey2, xy0s)
            ema_param = ema_kernel(ema_param, param, j, 500, 2, 0.99)
            print(f'CelebA{resolution} | {task} | {args.upsampling} | {args.sde} | {loss_type} | {args.schedule} | '
                  f'Epoch: {i} / {nepochs}, iter: {j} / {data_size // train_nsamples}, loss: {loss:.4f}')
        filename = f'./celeba{resolution}_{task}_{args.sde}_{args.schedule}_{i}.npz'
        if (i + 1) % 100 == 0:
            np.savez(filename, param=param, ema_param=ema_param)
else:
    param = np.load(f'./celeba{resolution}_{task}_{args.sde}_{args.schedule}_'
                    f'{args.test_epoch}.npz')['ema_param' if args.test_ema else 'param']


# Verify if the score function is learnt properly
def rev_sim(key_, u0):
    def learnt_score(x, t):
        return nn_score(x, t, param)

    return reverse_simulator(key_, u0, ts, learnt_score, sde.drift, sde.dispersion, integrator='euler-maruyama')


# Simulate the backward and verify if it matches the target distribution
key = jax.random.PRNGKey(args.test_seed)
key, subkey = jax.random.split(key)
test_sample0 = sampler(subkey, test=True)
key, subkey = jax.random.split(key)
traj = simulate_cond_forward(subkey, test_sample0, ts)
terminal_val = traj[-1]

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, None])(keys, terminal_val)
print(jnp.min(approx_init_samples), jnp.max(approx_init_samples))

fig, axes = plt.subplots(nrows=2, ncols=7, sharey='row')
for row in range(2):
    axes[row, 0].imshow(normalise_rgb(test_sample0[:, :, row * 3:(row + 1) * 3]))
    axes[row, 1].imshow(normalise_rgb(terminal_val[:, :, row * 3:(row + 1) * 3]))
    for i in range(2, 7):
        axes[row, i].imshow(normalise_rgb(approx_init_samples[i - 2][:, :, row * 3:(row + 1) * 3]))
plt.tight_layout(pad=0.1)
plt.savefig(f'./tmp_figs/celeba{resolution}_{task}_backward_test.png')
plt.show()

# Now conditional sampling
nparticles = args.nparticles
ngibbs = args.ngibbs


def reverse_drift(uv, t):
    return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * nn_score(uv, T - t, param)


def reverse_drift_u(u, v, t):
    rdu, rdv = dataset.unpack(reverse_drift(dataset.concat(u, v), t))
    return rdu


def reverse_drift_v(v, u, t):
    rdu, rdv = dataset.unpack(reverse_drift(dataset.concat(u, v), t))
    return rdv


def reverse_dispersion(t):
    return sde.dispersion(T - t)


def transition_sampler(us_prev, v_prev, t_prev, key_):
    return (us_prev + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us_prev, v_prev, t_prev) * dt
            + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                               u_prev + reverse_drift_u(u_prev, v_prev, t_prev) * dt,
                                               math.sqrt(dt) * reverse_dispersion(t_prev)))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def likelihood_logpdf(v, u_prev, v_prev, t_prev):
    cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
    return jnp.sum(jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * reverse_dispersion(t_prev)))


key, subkey = jax.random.split(key)
test_xy0 = sampler(subkey, test=True)
test_x0, test_y0 = test_xy0[:, :, :3], test_xy0[:, :, 3:]

fig, axes = plt.subplots(ncols=2)
axes[0].imshow(test_x0)
axes[1].imshow(test_y0)
plt.savefig(f'./tmp_figs/celeba{resolution}_{task}_pair.png')
plt.show()


def fwd_sampler(key_, x0):
    xy0 = dataset.concat(x0, test_y0)
    return simulate_cond_forward(key_, xy0, ts)


def bridge_sampler(key_, y0_, yT_):
    return doob_bridge_simulator(key_, sde, y0_, yT_, ts, integration_nsteps=100, replace=True)


@jax.jit
def gibbs_kernel(key_, xs_, us_star_, bs_star_):
    key_fwd, key_csmc, key_bridge = jax.random.split(key_, num=3)
    path_xy = fwd_sampler(key_fwd, xs_[0])
    path_x, path_y = dataset.unpack(path_xy)
    us = path_x[::-1]
    vs = bridge_sampler(key_fwd, path_y[0], path_y[-1])[::-1] if args.doob else path_y[::-1]

    def init_sampler(*_):
        return us[0] * jnp.ones((nparticles, *us.shape[1:]))

    def init_likelihood_logpdf(*_):
        return -math.log(nparticles) * jnp.ones(nparticles)

    us_star_next, bs_star_next = csmc_kernel(key_csmc,
                                             us_star_, bs_star_,
                                             vs, ts,
                                             init_sampler, init_likelihood_logpdf,
                                             transition_sampler, transition_logpdf,
                                             likelihood_logpdf,
                                             killing, nparticles,
                                             backward=True)
    xs_next = us_star_next[::-1]
    return xs_next, us_star_next, bs_star_next, bs_star_next != bs_star_


# Gibbs loop
key, subkey = jax.random.split(key)
xs = dataset.unpack(fwd_sampler(subkey, jnp.zeros_like(test_y0)))[0]
us_star = xs[::-1]
bs_star = jnp.zeros((nsteps + 1), dtype=int)

uss = np.zeros((ngibbs, nsteps + 1, resolution, resolution, 3))
for i in range(ngibbs):
    key, subkey = jax.random.split(key)
    xs, us_star, bs_star, acc = gibbs_kernel(subkey, xs, us_star, bs_star)
    uss[i] = us_star

    fig = plt.figure()
    plt.imshow(normalise_rgb(us_star[-1]))
    plt.tight_layout(pad=0.1)
    plt.savefig(f'./tmp_figs/celeba{resolution}_{task}_uss_{i}{"_doob" if args.doob else ""}.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(uss[:i, -1, 10, 10, 0])
    plt.tight_layout(pad=0.1)
    plt.savefig(f'./tmp_figs/celeba{resolution}_{task}_trace_pixel{"_doob" if args.doob else ""}.png')
    plt.close(fig)

    print(f'{task} | Gibbs iter: {i}, acc: {acc}')

np.save(f'uss{"_doob" if args.doob else ""}', uss)
