"""
Gaussian process regression with linear operator using filter.
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
import argparse
from fbs.samplers import bootstrap_filter, stratified
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=10, help='The problem dimension.')
parser.add_argument('--nparticles', type=int, default=10, help='The number of particles.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of samples to draw.')
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", False)

key = jax.random.PRNGKey(args.id)

# GP setting
ell, sigma = 1., 1.
d = args.d
zs = jnp.linspace(0., 5., d)
obs_var = 1.
H = jnp.diag(jnp.sin(jnp.linspace(0., 2. * jnp.pi, d)))


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
gp_mean = cov_mat @ jax.scipy.linalg.cho_solve(chol, y0)
gp_cov = cov_mat - cov_mat @ H.T @ jax.scipy.linalg.cho_solve(chol, H @ cov_mat)

joint_mean = jnp.zeros((2 * d,))
joint_cov = jnp.concatenate([jnp.concatenate([cov_mat, cov_mat @ H.T], axis=1),
                             jnp.concatenate([H @ cov_mat, H @ cov_mat @ H.T + obs_var * jnp.eye(d)], axis=1)],
                            axis=0)

# SDE noising process
T = 1.
nsteps = 200
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def forward_m_cov(t):
    F_, Q_ = discretise_linear_sde(t, ts[0])
    return F_ * joint_mean[:d], F_ ** 2 * cov_mat + Q_ * jnp.eye(d)


def score(z, t):
    mt, covt = forward_m_cov(t)
    chol = jax.scipy.linalg.cho_factor(covt)
    return -jax.scipy.linalg.cho_solve(chol, z - mt)


# Terminal reference distribution
m_ref, cov_ref = forward_m_cov(T)
chol_ref = jax.scipy.linalg.cho_factor(cov_ref)


# The reverse process
def reverse_drift(u, t):
    return -sde.drift(u, T - t) + sde.dispersion(T - t) ** 2 * score(u, T - t)


def reverse_dispersion(t):
    return sde.dispersion(T - t)


# Conditional sampling
nparticles = args.nparticles
nsamples = args.nsamples


def transition_sampler(us_prev, _, t_prev, key_):
    return (us_prev + jax.vmap(reverse_drift, in_axes=[0, None])(us_prev, t_prev) * dt
            + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def transition_logpdf(u, u_prev, v_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                               u_prev + reverse_drift(u_prev, t_prev) * dt,
                                               math.sqrt(dt) * reverse_dispersion(t_prev)))


@partial(jax.vmap, in_axes=[None, 0, None, None])
def likelihood_logpdf(v, u_prev, v_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(v, H @ u_prev, math.sqrt(dt) * reverse_dispersion(t_prev)))


def ref_sampler(key_, yT, nsamples_):
    return m_ref + jax.random.normal(key_, (nsamples_, d)) @ jnp.linalg.cholesky(cov_ref)


def fwd_ys_sampler(key_, y0_):
    def scan_body(carry, elem):
        y = carry
        t, t_prev, rnd = elem

        cov_diag_ = (1 - jnp.exp(-(t - t_prev))) * H @ H.T
        y = jnp.exp(-0.5 * (t - t_prev)) * y + jnp.sqrt(cov_diag_) @ rnd
        return y, y

    rnds_ = jax.random.normal(key_, shape=(nsteps, d))
    return jnp.concatenate([y0_[None, :], jax.lax.scan(scan_body, y0_, (ts[1:], ts[:-1], rnds_))[1]])


# Filter
@jax.jit
def conditional_sampler(key_):
    key_fwd, key_bwd, key_bf = jax.random.split(key_, num=3)
    path_y = fwd_ys_sampler(key_fwd, y0)
    vs = path_y[::-1]
    approx_x0 = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, ref_sampler, key_bf, nparticles,
                                 stratified, log=True, return_last=True)[0][0]
    return approx_x0


approx_cond_samples = np.zeros((nsamples, d))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    approx_cond_sample = conditional_sampler(subkey)
    approx_cond_samples[i] = approx_cond_sample
    print(f'ID: {args.id} | Sample {i}')

# Save results
np.savez(f'./toy/results/linear-filter-{args.sde}-{args.nparticles}-{args.id}',
         samples=approx_cond_samples, gp_mean=gp_mean, gp_cov=gp_cov)
