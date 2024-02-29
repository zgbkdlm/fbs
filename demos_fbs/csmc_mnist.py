r"""
Conditional sampling on MNIST
"""
import os
import math
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fbs.data import MNIST
from fbs.sdes import make_linear_sde, make_linear_sde_law_loss, StationaryConstLinearSDE, \
    StationaryLinLinearSDE, StationaryExpLinearSDE, reverse_simulator
from fbs.samplers import gibbs_init, gibbs_kernel
from fbs.nn.models import make_st_nn
from fbs.nn.unet import UNet
from fbs.nn.utils import make_optax_kernel
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST test.')
parser.add_argument('--train', action='store_true', default=False, help='Whether train or not.')
parser.add_argument('--task', type=str, default='deconv-10')
parser.add_argument('--sde', type=str, default='lin')
parser.add_argument('--upsampling', type=str, default='pixel_shuffle')
parser.add_argument('--loss_type', type=str, default='score')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--nsteps', type=int, default=2)
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--nepochs', type=int, default=30)
parser.add_argument('--save_mem', action='store_true', default=False,
                    help='Save memory by sharing the batch of x and t')
parser.add_argument('--grad_clip', action='store_true', default=False)
parser.add_argument('--test_nsteps', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=12)
parser.add_argument('--test_ema', action='store_true', default=False)
parser.add_argument('--test_seed', type=int, default=666)
parser.add_argument('--nparticles', type=int, default=100)
parser.add_argument('--ngibbs', type=int, default=10)
parser.add_argument('--ny0s', type=int, default=10)
parser.add_argument('--doob', action='store_true', default=False)

args = parser.parse_args()
train = args.train
task = args.task

print(f'{"Train" if train else "Test"} {task} on MNIST')

# General configs
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)

T = 1
nsteps = args.test_nsteps
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

# MNIST
d = (28, 28, 2)
key, subkey = jax.random.split(key)
dataset = MNIST(subkey, '../datasets/mnist.npz', task=task)
dataset_test = MNIST(subkey, '../datasets/mnist.npz', task=task, test=True)


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
            axes[row, col].imshow(xys[col, :, :, row], cmap='gray')
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
my_nn = UNet(dt=nn_dt, dim=32, upsampling=args.upsampling)
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
        for j in range(data_size // train_nsamples):
            subkey, subkey2 = jax.random.split(subkey)
            x0s, y0s = dataset.enumerate_subset(j, perm_inds, subkey)
            xy0s = dataset.concat(x0s, y0s)
            param, opt_state, loss = optax_kernel(param, opt_state, subkey2, xy0s)
            ema_param = ema_kernel(ema_param, param, j, 500, 2, 0.99)
            print(f'MNIST | {task} | {args.upsampling} | {args.sde} | {loss_type} | {args.schedule} | '
                  f'Epoch: {i} / {nepochs}, iter: {j} / {data_size // train_nsamples}, loss: {loss:.4f}')
        filename = f'./mnist_{task}_{args.sde}_{args.schedule}_{i}.npz'
        if (i + 1) % 100 == 0:
            np.savez(filename, param=param, ema_param=ema_param)
else:
    param = np.load(f'./mnist_{task}_{args.sde}_{args.schedule}_'
                    f'{args.test_epoch}.npz')['ema_param' if args.test_ema else 'param']


# Verify if the score function is learnt properly
def rev_sim(key_, u0):
    def learnt_score(x, t):
        return nn_score(x, t, param)

    return reverse_simulator(key_, u0, ts, learnt_score, sde.drift, sde.dispersion, integrator='euler-maruyama')


# Simulate the backward and verify if it matches the target distribution
kkk = jax.random.PRNGKey(args.test_seed)
key, subkey = jax.random.split(kkk)
test_x0 = sampler(subkey, test=True)
key, subkey = jax.random.split(key)
traj = simulate_cond_forward(subkey, test_x0, ts)
terminal_val = traj[-1]

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=5)
approx_init_samples = jax.vmap(rev_sim, in_axes=[0, None])(keys, terminal_val)
print(jnp.min(approx_init_samples), jnp.max(approx_init_samples))

fig, axes = plt.subplots(nrows=2, ncols=7, sharey='row')
for row in range(2):
    axes[row, 0].imshow(test_x0[:, :, row], cmap='gray')
    axes[row, 1].imshow(terminal_val[:, :, row], cmap='gray')
    for i in range(2, 7):
        axes[row, i].imshow(approx_init_samples[i - 2][:, :, row], cmap='gray')
plt.tight_layout(pad=0.1)
plt.savefig(f'./tmp_figs/mnist_{task}_backward_test.png')
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


def fwd_sampler(key_, x0_, y0_):
    xy0 = dataset.concat(x0_, y0_)
    return simulate_cond_forward(key_, xy0, ts)


gibbs_kernel = jax.jit(partial(gibbs_kernel, ts=ts, fwd_sampler=fwd_sampler, sde=sde, dataset=dataset,
                               nparticles=nparticles, transition_sampler=transition_sampler,
                               transition_logpdf=transition_logpdf, likelihood_logpdf=likelihood_logpdf, doob=True))
gibbs_init = jax.jit(partial(gibbs_init, x0_shape=(28, 28, 1), ts=ts, fwd_sampler=fwd_sampler, dataset=dataset,
                             transition_sampler=transition_sampler, transition_logpdf=transition_logpdf,
                             likelihood_logpdf=likelihood_logpdf,
                             nparticles=nparticles, method='smoother'))

for k in range(args.ny0s):
    print(f'Running Gibbs sampler for {k}-th y0.')
    key, subkey = jax.random.split(key)
    test_xy0 = sampler(subkey, test=True)
    test_x0, test_y0 = test_xy0[:, :, :1], test_xy0[:, :, 1:]

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(test_x0, cmap='gray')
    axes[1].imshow(test_y0, cmap='gray')
    plt.savefig(f'./tmp_figs/mnist_{task}_pair_{k}.png')
    plt.show()

    # Gibbs loop
    key, subkey = jax.random.split(key)
    x0, us_star = gibbs_init(subkey, test_y0)
    bs_star = jnp.zeros((nsteps + 1), dtype=int)

    fig = plt.figure()
    plt.imshow(x0[:, :, 0], cmap='gray')
    plt.tight_layout(pad=0.1)
    plt.savefig(f'./tmp_figs/mnist_{task}_init_{k}.png')
    plt.close(fig)

    for i in range(ngibbs):
        key, subkey = jax.random.split(key)
        x0, us_star, bs_star, acc = gibbs_kernel(subkey, x0, test_y0, us_star, bs_star)

        fig = plt.figure()
        plt.imshow(us_star[-1, :, :, 0], cmap='gray')
        plt.tight_layout(pad=0.1)
        plt.savefig(f'./tmp_figs/mnist_{task}_uss_{i}{"_doob" if args.doob else ""}_{k}.png')
        plt.close(fig)

        print(f'{task} | Gibbs iter: {i}, acc: {acc}')
