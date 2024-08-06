"""
Gaussian process regression with linear operator using diffusion Gibbs.
"""
import jax
import jax.numpy as jnp
import math
import numpy as np
import argparse
from fbs.samplers import bootstrap_filter_aug as bootstrap_filter, stratified, gibbs_kernel_aug as gibbs_kernel_
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=100, help='The problem dimension.')
parser.add_argument('--nparticles', type=int, default=10, help='The number of particles.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of samples to draw.')
parser.add_argument('--explicit_backward', action='store_true', default=False,
                    help='Whether to explicitly sample the CSMC backward')
parser.add_argument('--explicit_final', action='store_true', default=False,
                    help='Whether to ue ref in CSMC.')
parser.add_argument('--marg', action='store_true', default=False, help='Whether marginalise out the Y path.')
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
parser.add_argument('--obs_var', type=float, default=1, help='The observation noise variance.')
parser.add_argument('--nchains', type=int, default=1, help='The number of MCMC chains.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", False)

key = jax.random.PRNGKey(args.id)

# GP setting
ell, sigma = 1., 1.
d = args.d
zs = jnp.linspace(0., 5., d)
obs_var = args.obs_var
H = jnp.diag(jnp.linspace(-2, 2, d))


def cov_fn(z1, z2):
    return sigma ** 2 * jnp.exp(-jnp.abs(z1[None, :] - z2[:, None]) / ell)


# Generate a y0
key, subkey = jax.random.split(key)
fs = jnp.linalg.cholesky(cov_fn(zs, zs)) @ jax.random.normal(subkey, (d,))
key, subkey = jax.random.split(key)
y0 = H @ fs + jnp.sqrt(obs_var) * jax.random.normal(subkey, (d,))

# GP regression
cov_mat = cov_fn(zs, zs)
chol = jax.scipy.linalg.cho_factor(H @ cov_mat @ H.T + obs_var * jnp.eye(d))
gp_mean = cov_mat @ H.T @ jax.scipy.linalg.cho_solve(chol, y0)
gp_cov = cov_mat - cov_mat @ H.T @ jax.scipy.linalg.cho_solve(chol, H @ cov_mat)

joint_mean = jnp.zeros((2 * d,))
joint_cov = jnp.concatenate([jnp.concatenate([cov_mat, cov_mat @ H.T], axis=1),
                             jnp.concatenate([H @ cov_mat, H @ cov_mat @ H.T + obs_var * jnp.eye(d)], axis=1)],
                            axis=0)
H_ = jnp.concatenate([jnp.eye(d), H], axis=0)

# SDE noising process
T = 1.
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

USE_SVD = True
U, D, V = jnp.linalg.svd(cov_mat)

joint_mT, joint_vT = jnp.exp(-0.5 * T) * joint_mean, jnp.exp(-T) * joint_cov + (1 - jnp.exp(-T)) * H_ @ H_.T
chol_ref = jax.scipy.linalg.cho_factor(joint_vT[d:, d:])
cond_cov_ref = joint_vT[:d, :d] - joint_vT[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref, joint_vT[d:, :d])
cond_chol_ref = jnp.linalg.cholesky(cond_cov_ref)


def forward_m_cov(t):
    F_, Q_ = discretise_linear_sde(t, ts[0])
    return F_ * joint_mean[:d], F_ ** 2 * cov_mat + Q_ * jnp.eye(d)


def forward_m_svd(t):
    F_, Q_ = discretise_linear_sde(t, ts[0])
    cov_t_D = F_ ** 2 * D + Q_ * jnp.ones((d,))
    return F_ * joint_mean[:d], cov_t_D


def score(z, t):
    if USE_SVD:
        mt, cov_t_D = forward_m_svd(t)
        cov_t_D_inv = 1. / cov_t_D  # shape (d,)
        tmp = V @ (z - mt)
        out = -U @ (tmp * cov_t_D_inv)
    else:
        mt, covt = forward_m_cov(t)
        cholt = jax.scipy.linalg.cho_factor(covt)
        out = -jax.scipy.linalg.cho_solve(cholt, z - mt)
    return out


def unpack(xy):
    return xy[..., :d], xy[..., d:]


# The reverse process
def reverse_drift(u, t):
    return -sde.drift(u, T - t) + sde.dispersion(T - t) ** 2 * score(u, T - t)


def reverse_dispersion(t):
    return sde.dispersion(T - t)


# Conditional sampling
nparticles = args.nparticles
nsamples = args.nsamples
nchains = args.nchains
chain_track_id = 0
burnin = 100


def transition_sampler(us_prev, v_prev, t_prev, key_):
    return (us_prev + jax.vmap(reverse_drift, in_axes=[0, None])(us_prev, t_prev) * dt
            + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                               u_prev + reverse_drift(u_prev, t_prev) * dt,
                                               math.sqrt(dt) * reverse_dispersion(t_prev)))


@partial(jax.vmap, in_axes=[None, 0, None])
def likelihood_logpdf(v, u, t):
    scale = obs_var ** 0.5 * jnp.exp(-0.5 * (T - t))
    return jnp.sum(jax.scipy.stats.norm.logpdf(v, H @ u, scale))


def ref_sampler(key_, yT, nsamples_):
    cond_m_ = joint_mT[:d] + joint_vT[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref, yT - joint_mT[d:])
    return cond_m_ + jax.random.normal(key_, (nsamples_, d)) @ cond_chol_ref


def fwd_sampler(key_, x0_, y0_):
    def scan_body(carry, elem):
        x, y = carry
        t, t_prev, rnd = elem

        cov_diag_ = (1 - jnp.exp(-(t - t_prev)))
        x = jnp.exp(-0.5 * (t - t_prev)) * x + jnp.sqrt(cov_diag_) * rnd
        y = jnp.exp(-0.5 * (t - t_prev)) * y + jnp.sqrt(cov_diag_) * H @ rnd
        return (x, y), (x, y)

    rnds_ = jax.random.normal(key_, shape=(nsteps, d))
    xs_, ys_ = jax.lax.scan(scan_body, (x0_, y0_), (ts[1:], ts[:-1], rnds_))[1]
    return jnp.concatenate([jnp.concatenate([x0_[None, :], xs_], axis=0),
                            jnp.concatenate([y0_[None, :], ys_], axis=0)], axis=1)


def fwd_ys_sampler(key_, y0_):
    def scan_body(carry, elem):
        y = carry
        t, t_prev, rnd = elem

        cov_diag_ = (1 - jnp.exp(-(t - t_prev)))
        y = jnp.exp(-0.5 * (t - t_prev)) * y + jnp.sqrt(cov_diag_) * H @ rnd
        return y, y

    rnds_ = jax.random.normal(key_, shape=(nsteps, d))
    return jnp.concatenate([y0_[None, :], jax.lax.scan(scan_body, y0_, (ts[1:], ts[:-1], rnds_))[1]])


# Gibbs initial
def gibbs_init(key_):
    key_fwd, key_bwd, key_bf = jax.random.split(key_, num=3)
    path_y = fwd_ys_sampler(key_fwd, y0)
    vs = path_y[::-1]
    uss = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, ref_sampler, key_bf, nparticles,
                           stratified, log=True, return_last=False)[0]
    x0 = uss[-1, 0]
    bs_star = jnp.zeros((nsteps + 1), dtype=int)
    return x0, bs_star


# Gibbs kernel
gibbs_kernel = partial(gibbs_kernel_, ts=ts, fwd_sampler=fwd_sampler, sde=sde, unpack=unpack,
                       nparticles=nparticles, transition_sampler=transition_sampler,
                       transition_logpdf=transition_logpdf, likelihood_logpdf=likelihood_logpdf,
                       marg_y=args.marg,
                       explicit_backward=args.explicit_backward, explicit_final=args.explicit_final)

gibbs_init_chain_vmap = jax.vmap(gibbs_init, in_axes=[0])
gibbs_kernel_chain_vmap = jax.jit(jax.vmap(gibbs_kernel, in_axes=[0, 0, None, 0, 0]))

# Gibbs loop
key, subkey = jax.random.split(key)
key_chains = jax.random.split(subkey, num=nchains)
x0s, bs_stars = gibbs_init_chain_vmap(key_chains)

gibbs_samples = np.zeros((nchains, nsamples, d))
accs = np.zeros((nsamples,), dtype=bool)
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    key_chains = jax.random.split(subkey, num=nchains)
    x0s, _, bs_stars, acc = gibbs_kernel_chain_vmap(key_chains, x0s, y0, None, bs_stars)
    gibbs_samples[:, i, :] = x0s
    accs[i] = acc[chain_track_id, -1]
    j = max(0, i - 100)
    print(f'ID: {args.id} | obs_var: {obs_var} | Gibbs | iter: {i} | acc : {acc[chain_track_id, -1]} | '
          f'acc rate: {np.mean(accs[:i]):.3f} | acc rate last 100: {np.mean(accs[j:i]):.3f}')

# # Save results
# np.savez(f'./toy/results/linear-gibbs{"-eb" if args.explicit_backward else ""}{"-ef" if args.explicit_final else ""}'
#          f'{"-marg" if args.marg else ""}-{args.nparticles}-{args.id}-{args.obs_var}',
#          samples=gibbs_samples, gp_mean=gp_mean, gp_cov=gp_cov)

# Plot
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
axes[0].plot(zs, gp_mean, linewidth=2, linestyle='--', c='black', label='GP mean')
axes[0].plot(zs, np.mean(gibbs_samples[0], axis=0), linewidth=2, linestyle='-', c='black', label='PF approx. mean')
axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].legend()
mesh_ = np.meshgrid(zs, zs)
residual = np.abs(np.cov(gibbs_samples[0], rowvar=False) - gp_cov)
print(np.max(residual))
axes[1].pcolormesh(*mesh_, residual, cmap=plt.cm.binary, vmin=0, vmax=0.5)
axes[1].set_title('Absolute difference between the approx. and true GP covariances')
plt.tight_layout(pad=0.1)
plt.show()
