"""
Gaussian process regression using Gibbs CSMC.
"""
import jax
import jax.numpy as jnp
import math
import numpy as np
import argparse
from fbs.samplers import bootstrap_filter, stratified
from fbs.samplers.smc import pmcmc_kernel
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=10, help='The problem dimension.')
parser.add_argument('--nparticles', type=int, default=10, help='The number of particles.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of samples to draw.')
parser.add_argument('--sde', type=str, default='const', help='The type of forward SDE.')
parser.add_argument('--delta', type=float, default=0.01, help='The delta value of pMCMC')
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
parser.add_argument('--nchains', type=int, default=4, help='The number of MCMC chains.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", False)

key = jax.random.PRNGKey(args.id)

# GP setting
ell, sigma = 1., 1.
d = args.d
zs = jnp.linspace(0., 5., d)
obs_var = 1.


def cov_fn(z1, z2):
    return sigma ** 2 * jnp.exp(-jnp.abs(z1[None, :] - z2[:, None]) / ell)


# Generate a y0
key, subkey = jax.random.split(key)
fs = jnp.linalg.cholesky(cov_fn(zs, zs)) @ jax.random.normal(subkey, (d,))
key, subkey = jax.random.split(key)
y0 = fs + jnp.sqrt(obs_var) * jax.random.normal(subkey, (d,))

# GP regression
cov_mat = cov_fn(zs, zs)
chol = jax.scipy.linalg.cho_factor(cov_mat + obs_var * jnp.eye(d))
gp_mean = cov_mat @ jax.scipy.linalg.cho_solve(chol, y0)
gp_cov = cov_mat - cov_mat @ jax.scipy.linalg.cho_solve(chol, cov_mat)

joint_mean = jnp.zeros((2 * d,))
joint_cov = jnp.concatenate([jnp.concatenate([cov_mat, cov_mat], axis=1),
                             jnp.concatenate([cov_mat, cov_mat + obs_var * jnp.eye(d)], axis=1)],
                            axis=0)

# SDE noising process
T = 1.
nsteps = 200
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

if args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=4., t0=0., T=T)
else:
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def forward_m_cov(t):
    F_, Q_ = discretise_linear_sde(t, ts[0])
    return F_ * joint_mean, F_ ** 2 * joint_cov + Q_ * jnp.eye(2 * d)


def score(z, t):
    mt, covt = forward_m_cov(t)
    chol = jax.scipy.linalg.cho_factor(covt)
    return -jax.scipy.linalg.cho_solve(chol, z - mt)


# Terminal reference distribution
m_ref, cov_ref = forward_m_cov(T)
chol_ref = jax.scipy.linalg.cho_factor(cov_ref[d:, d:])


# The reverse process
def reverse_drift(uv, t):
    return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * score(uv, T - t)


def reverse_drift_u(u, v, t):
    uv = jnp.concatenate([u, v])
    return reverse_drift(uv, t)[:d]


def reverse_drift_v(v, u, t):
    uv = jnp.concatenate([u, v])
    return reverse_drift(uv, t)[d:]


def reverse_dispersion(t):
    return sde.dispersion(T - t)


# Conditional sampling
nparticles = args.nparticles
nsamples = args.nsamples
nchains = args.nchains
chain_track_id = 0
burnin = 100


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


def ref_sampler(key_, yT, nsamples_):
    m_ = m_ref[:d] + cov_ref[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref, yT - m_ref[d:])
    cov_ = cov_ref[:d, :d] - cov_ref[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref, cov_ref[d:, :d])
    return m_ + jax.random.normal(key_, (nsamples_, d)) @ jnp.linalg.cholesky(cov_)


def fwd_sampler(key_, x0_, y0_):
    return simulate_cond_forward(key_, jnp.concatenate([x0_, y0_]), ts)


def fwd_ys_sampler(key_, y0_):
    return simulate_cond_forward(key_, y0_, ts)


# pMCMC initial
def pmcmc_init(key_):
    key_fwd, key_bwd, key_bf, key_ys = jax.random.split(key_, num=4)
    path_y = fwd_ys_sampler(key_fwd, y0)
    vs = path_y[::-1]
    x0s, log_ell = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, ref_sampler, key_bf, nparticles,
                                    stratified, log=True, return_last=True)
    return x0s[0], log_ell, fwd_ys_sampler(key_ys, y0)


# pMCMC kernel
pmcmc_kernel = partial(pmcmc_kernel, ts=ts, fwd_ys_sampler=fwd_ys_sampler, sde=sde,
                       ref_sampler=ref_sampler, transition_sampler=transition_sampler,
                       likelihood_logpdf=likelihood_logpdf, resampling=stratified,
                       nparticles=nparticles, delta=args.delta)

pmcmc_init_chain_vmap = jax.vmap(pmcmc_init, in_axes=[0])
pmcmc_kernel_chain_vmap = jax.jit(jax.vmap(pmcmc_kernel, in_axes=[0, 0, 0, 0, None]))

# pMCMC loop
key, subkey = jax.random.split(key)
key_chains = jax.random.split(subkey, num=nchains)
x0s, log_ells, yss = pmcmc_init_chain_vmap(key_chains)

pmcmc_samples = np.zeros((nchains, nsamples, d))
accs = np.zeros((nsamples,))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    key_chains = jax.random.split(subkey, num=nchains)
    x0s, log_ells, yss, mcmc_states = pmcmc_kernel_chain_vmap(key_chains, x0s, log_ells, yss, y0)
    pmcmc_samples[:, i, :] = x0s
    accs[i] = mcmc_states.acceptance_prob[chain_track_id]
    j = max(0, i - 100)
    print(f'ID: {args.id} | pMCMC | iter: {i} | '
          f'acc_prob: {mcmc_states.acceptance_prob[chain_track_id]:.3f} '
          f'| {np.mean(accs[:i]):.3f} | {np.mean(accs[j:i]):.3f}')

# Save results
np.savez(f'./toy/results/pmcmc-{args.delta}-{args.sde}-{args.nparticles}-{args.id}',
         samples=pmcmc_samples, gp_mean=gp_mean, gp_cov=gp_cov)
