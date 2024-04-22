"""
Gaussian process regression using filter. Ablation study of the effect of the correlation in the diffusion.
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
import argparse
from fbs.samplers import bootstrap_filter, stratified
from fbs.sdes import make_gaussian_bw_sb, euler_maruyama
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=10, help='The problem dimension.')
parser.add_argument('--nparticles', type=int, default=10, help='The number of particles.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of samples to draw.')
parser.add_argument('--x0', type=str, default='proper', help='How to draw x0.')
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", False)

key = jax.random.PRNGKey(args.id)

# GP setting
ell, sigma = 1., 1.
d = args.d
zs = jnp.linspace(0., 5., d)
obs_var = 0.1


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
gp_posterior_mean = cov_mat @ jax.scipy.linalg.cho_solve(chol, y0)
gp_posterior_cov = cov_mat - cov_mat @ jax.scipy.linalg.cho_solve(chol, cov_mat)
gp_posterior_chol = jnp.linalg.cholesky(gp_posterior_cov)


def gp_posterior_sampler(key_):
    return gp_posterior_mean + jax.random.normal(key_, (d,)) @ gp_posterior_chol


joint_mean = jnp.zeros((2 * d,))
joint_cov = jnp.concatenate([jnp.concatenate([cov_mat, cov_mat], axis=1),
                             jnp.concatenate([cov_mat, cov_mat + obs_var * jnp.eye(d)], axis=1)],
                            axis=0)

# Reference distribution
ref_m = jnp.ones((2 * d,))
key, subkey = jax.random.split(key)
a_ = jax.random.normal(subkey, (2 * d, 2 * d))
ref_cov = a_ @ a_.T
chol_ref_y = jax.scipy.linalg.cho_factor(ref_cov[d:, d:])

# The Schrodinger bridge
T = 1.
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0, T, nsteps + 1)

marginal_mean, marginal_cov, drift = make_gaussian_bw_sb(joint_mean, joint_cov, ref_m, ref_cov, sig=1.)


def dispersion(_):
    return 1.


def score(z, t):
    mt, covt = marginal_mean(t), marginal_cov(t)
    chol = jax.scipy.linalg.cho_factor(covt)
    return -jax.scipy.linalg.cho_solve(chol, z - mt)


# The reverse process
def reverse_drift(uv, t):
    return -drift(uv, T - t) + dispersion(T - t) ** 2 * score(uv, T - t)


def reverse_drift_u(u, v, t):
    uv = jnp.concatenate([u, v])
    return reverse_drift(uv, t)[:d]


def reverse_drift_v(v, u, t):
    uv = jnp.concatenate([u, v])
    return reverse_drift(uv, t)[d:]


def reverse_dispersion(t):
    return dispersion(T - t)


# Conditional sampling
nparticles = args.nparticles
nsamples = args.nsamples


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
    posterior_ref_m = ref_m[:d] + ref_cov[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref_y, yT - ref_m[d:])
    posterior_ref_cov = ref_cov[:d, :d] - ref_cov[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref_y, ref_cov[d:, :d])
    return posterior_ref_m + jax.random.normal(key_, (nsamples_, d)) @ jnp.linalg.cholesky(posterior_ref_cov)


def fwd_ys_sampler_proper(key_):
    key_x0, key_em = jax.random.split(key_)
    x0_ = gp_posterior_sampler(key_x0)
    xy0 = jnp.concatenate([x0_, y0])
    return euler_maruyama(key_em, xy0, ts, drift, dispersion, integration_nsteps=10, return_path=True)[:, d:]


def fwd_ys_sampler_heuristic(key_):
    key_x0, key_em = jax.random.split(key_)
    x0_ = jax.random.normal(key_x0, (d,))
    xy0 = jnp.concatenate([x0_, y0])
    return euler_maruyama(key_em, xy0, ts, drift, dispersion, integration_nsteps=10, return_path=True)[:, d:]


# Filter
@jax.jit
def conditional_sampler(key_):
    key_fwd, key_bwd, key_bf = jax.random.split(key_, num=3)
    if args.x0 == 'proper':
        path_y = fwd_ys_sampler_proper(key_fwd)
    elif args.x0 == 'heuristic':
        path_y = fwd_ys_sampler_heuristic(key_fwd)
    else:
        raise ValueError(f'Invalid "{args.x0}" method')
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
np.savez(f'./sb/results/filter-{args.nparticles}-{args.x0}-{args.id}',
         samples=approx_cond_samples, gp_mean=gp_posterior_mean, gp_cov=gp_posterior_cov)
