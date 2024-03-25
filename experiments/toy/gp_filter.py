"""
Gaussian process regression using filter.
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from fbs.samplers import bootstrap_filter, stratified
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, reverse_simulator
from fbs.utils import bures_dist
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(args.id)

# GP setting
ell, sigma = 1., 1.
d = 10
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
nsteps = 500
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

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


def rev_sim(key_, uv0):
    return reverse_simulator(key_, uv0, ts, score, sde.drift, sde.dispersion,
                             integrator='euler-maruyama', integration_nsteps=10)


# Conditional sampling
nparticles = 10
nsamples = 1000


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


def fwd_ys_sampler(key_, y0_):
    return simulate_cond_forward(key_, y0_, ts)


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
    print(f'Sample {i}')

# Save results
np.save(f'./toy/results/gp-filter-{args.id}', approx_cond_samples)

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
