"""
Gaussian process regression using twisted SMC.
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from fbs.samplers import twisted_smc, stratified
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE
from fbs.utils import bures_dist
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=100, help='The problem dimension.')
parser.add_argument('--nparticles', type=int, default=10, help='The number of particles.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of samples to draw.')
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)

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

plt.plot(zs, fs)
plt.scatter(zs, y0, s=1)
plt.plot(zs, gp_mean)
plt.fill_between(zs,
                 gp_mean - 1.96 * jnp.sqrt(jnp.diag(gp_cov)),
                 gp_mean + 1.96 * jnp.sqrt(jnp.diag(gp_cov)),
                 alpha=0.3, color='black', edgecolor='none')
plt.show()

# SDE noising process
T = 1.
nsteps = 200
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def forward_m_cov(t):
    F_, Q_ = discretise_linear_sde(t, ts[0])
    return F_ * joint_mean, F_ ** 2 * joint_cov + Q_ * jnp.eye(2 * d)


def score(u, t):
    mt, covt = forward_m_cov(t)
    mt, covt = mt[:d], covt[:d, :d]
    chol = jax.scipy.linalg.cho_factor(covt)
    return -jax.scipy.linalg.cho_solve(chol, u - mt)


# Terminal reference distribution
m_ref, cov_ref = forward_m_cov(T)


# The reverse process
def reverse_drift(u, t):
    return -sde.drift(u, T - t) + sde.dispersion(T - t) ** 2 * score(u, T - t)


def reverse_cond_drift(u, t, y):
    return reverse_drift(u, t) + jax.grad(twisting_logpdf, argnums=1)(y, u, t)


def reverse_dispersion(t):
    return sde.dispersion(T - t)


# Conditional sampling
nparticles = args.nparticles
nsamples = args.nsamples


@partial(jax.vmap, in_axes=[0, 0, None])
def transition_logpdf(u, u_prev, t_prev):
    return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                               u_prev + reverse_drift(u_prev, t_prev) * dt,
                                               math.sqrt(dt) * reverse_dispersion(t_prev)))


def init_sampler(key_, nparticles_):
    return m_ref[:d] + jnp.einsum('ij,nj->ni',
                                  jnp.linalg.cholesky(cov_ref[:d, :d]),
                                  jax.random.normal(key_, (nparticles_, d)))


def twisting_logpdf(y, u, t):
    denoising_estimate = u + score(u, T - t) * dt
    return jnp.sum(jax.scipy.stats.norm.logpdf(y, denoising_estimate, jnp.sqrt(obs_var)))


twisting_logpdf_vmap = jax.vmap(twisting_logpdf, in_axes=[None, 0, None])


def twisting_prop_sampler(key_, us, t, y):
    m_ = us + jax.vmap(reverse_cond_drift, in_axes=[0, None, None])(us, t, y) * dt
    return m_ + math.sqrt(dt) * reverse_dispersion(t) * jax.random.normal(key_, (nparticles, d))


@partial(jax.vmap, in_axes=[0, 0, None, None])
def twisting_prop_logpdf(u, u_prev, t, y):
    m_ = u_prev + reverse_cond_drift(u_prev, t, y) * dt
    return jnp.sum(jax.scipy.stats.norm.logpdf(u, m_, math.sqrt(dt) * reverse_dispersion(t)))


# Filter
@jax.jit
def conditional_sampler(key_):
    key_filter, key_select = jax.random.split(key_)
    uvs, log_ws = twisted_smc(key_filter, y0, ts,
                              init_sampler, transition_logpdf, twisting_logpdf_vmap, twisting_prop_sampler,
                              twisting_prop_logpdf,
                              resampling=stratified, nparticles=nparticles)
    return jax.random.choice(key_select, uvs, p=jnp.exp(log_ws), axis=0)


approx_cond_samples = np.zeros((nsamples, d))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    approx_cond_sample = conditional_sampler(subkey)
    approx_cond_samples[i] = approx_cond_sample
    print(f'ID: {args.id} | Sample {i}')

# Plot
approx_gp_mean = jnp.mean(approx_cond_samples, axis=0)
approx_gp_cov = jnp.cov(approx_cond_samples, rowvar=False)
distance = bures_dist(gp_mean, gp_cov, approx_gp_mean, approx_gp_cov)
print(f'Bures distance {distance}')

plt.plot(zs, gp_mean)
plt.fill_between(zs,
                 gp_mean - 1.96 * jnp.sqrt(jnp.diag(gp_cov)),
                 gp_mean + 1.96 * jnp.sqrt(jnp.diag(gp_cov)),
                 alpha=0.3, color='black', edgecolor='none')
plt.plot(zs, approx_gp_mean)
plt.fill_between(zs,
                 approx_gp_mean - 1.96 * jnp.sqrt(jnp.diag(approx_gp_cov)),
                 approx_gp_mean + 1.96 * jnp.sqrt(jnp.diag(approx_gp_cov)),
                 alpha=0.3, color='tab:red', edgecolor='none')
plt.show()
