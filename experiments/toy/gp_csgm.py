"""
Gaussian process regression using exact conditional score matching, i.e., https://arxiv.org/pdf/2011.13456.pdf.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import argparse
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE
from fbs.sdes.simulators import euler_maruyama

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=100, help='The problem dimension.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of samples to draw.')
parser.add_argument('--sde', type=str, default='const', help='The type of forward SDE.')
parser.add_argument('--id', type=int, default=666, help='The id of independent MC experiment.')
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
chol_gp = jax.scipy.linalg.cho_factor(cov_mat + obs_var * jnp.eye(d))
gp_mean = cov_mat @ jax.scipy.linalg.cho_solve(chol_gp, y0)
gp_cov = cov_mat - cov_mat @ jax.scipy.linalg.cho_solve(chol_gp, cov_mat)

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

if args.sde == 'lin':
    sde = StationaryLinLinearSDE(beta_min=0.02, beta_max=4., t0=0., T=T)
else:
    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)


def forward_m_cov(t):
    F_, Q_ = discretise_linear_sde(t, ts[0])
    return F_ * joint_mean[:d], F_ ** 2 * cov_mat + Q_ * jnp.eye(d)


# Terminal reference distribution
F_ref, Q_ref = discretise_linear_sde(T, ts[0])
cond_m_ref = F_ref * joint_mean[:d] + F_ref * cov_mat @ jax.scipy.linalg.cho_solve(chol_gp, y0 - joint_mean[:d])
cond_cov_ref = F_ref ** 2 * cov_mat + Q_ref * jnp.eye(d) - F_ref * cov_mat @ jax.scipy.linalg.cho_solve(chol_gp,
                                                                                                        F_ref * cov_mat)


def cond_ref_sampler(key_):
    return cond_m_ref + cond_cov_ref @ jax.random.normal(key_, (d,))


# The reverse process
def reverse_drift(u, t):
    F, Q = discretise_linear_sde(T - t, ts[0])
    chol = jax.scipy.linalg.cho_factor(F ** 2 * cov_mat + Q * jnp.eye(d))

    score_x = -jax.scipy.linalg.cho_solve(chol, u - F * joint_mean[:d])

    def cond_logpdf(x_):
        cond_m = joint_mean[:d] + cov_mat * F @ jax.scipy.linalg.cho_solve(chol, x_ - F * joint_mean[:d])
        cond_cov = cov_mat + obs_var * jnp.eye(d) - cov_mat * F @ jax.scipy.linalg.cho_solve(chol, F * cov_mat)
        return jax.scipy.stats.multivariate_normal.logpdf(y0, cond_m, cond_cov)

    return -sde.drift(u, T - t) + sde.dispersion(T - t) ** 2 * (score_x + jax.grad(cond_logpdf)(u))


def reverse_dispersion(t):
    return sde.dispersion(T - t)


# Conditional sampling
nsamples = args.nsamples


@jax.jit
def conditional_sampler(key_):
    key_init, key_sde = jax.random.split(key_, num=2)
    u0 = cond_ref_sampler(key_init)
    uT = euler_maruyama(key_sde, u0, ts, reverse_drift, reverse_dispersion,
                        integration_nsteps=1, return_path=False)
    return uT


approx_cond_samples = np.zeros((nsamples, d))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    approx_cond_sample = conditional_sampler(subkey)
    approx_cond_samples[i] = approx_cond_sample
    print(f'ID: {args.id} | Sample {i}')

# Save results
np.savez(f'./toy/results/csgm-{args.sde}-{args.id}',
         samples=approx_cond_samples, gp_mean=gp_mean, gp_cov=gp_cov)

# Plot
approx_gp_mean = jnp.mean(approx_cond_samples, axis=0)
approx_gp_cov = jnp.cov(approx_cond_samples, rowvar=False)

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
