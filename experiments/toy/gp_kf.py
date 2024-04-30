"""
Quickly test the KF approach. Not compared, since it should be exact.
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=10, help='The problem dimension.')
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
nsteps = 2
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
nsamples = args.nsamples


def cond_state_mean(u_prev, v_prev, t_prev):
    return u_prev + reverse_drift_u(u_prev, v_prev, t_prev) * dt


def cond_state_cov(t_prev):
    return math.sqrt(dt) * reverse_dispersion(t_prev) * jnp.eye(d)


def cond_obs_mean(u_prev, v_prev, t_prev):
    return v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt


def cond_obs_cov(t_prev):
    return math.sqrt(dt) * reverse_dispersion(t_prev) * jnp.eye(d)


def ref_solver(yT):
    m_ = m_ref[:d] + cov_ref[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref, yT - m_ref[d:])
    cov_ = cov_ref[:d, :d] - cov_ref[:d, d:] @ jax.scipy.linalg.cho_solve(chol_ref, cov_ref[d:, :d])
    return m_, cov_


def fwd_ys_sampler(key_, y0_):
    return simulate_cond_forward(key_, y0_, ts)


def adhoc_kf(init_m, init_v, vs):
    def scan_body(carry, elem):
        mf, vf = carry
        v, v_prev, t_prev = elem

        F = jax.jacfwd(cond_state_mean, argnums=0)(mf, v_prev, t_prev)
        mp = cond_state_mean(mf, v_prev, t_prev)
        vp = F @ vf @ F.T + cond_state_cov(t_prev)

        H = jax.jacfwd(cond_obs_mean, argnums=1)(mp, v_prev, t_prev)
        S = H @ vp @ H.T + cond_obs_cov(t_prev)
        chol_s = jax.scipy.linalg.cho_factor(S)
        K = vp @ jax.scipy.linalg.cho_solve(chol_s, H).T
        mf = mp + K @ (v - cond_obs_mean(mp, v_prev, t_prev))
        vf = vp - K @ S @ K.T
        return (mf, vf), None

    return jax.lax.scan(scan_body, (init_m, init_v), (vs[1:], vs[:-1], ts[:-1]))[0]


@jax.jit
def conditional_sampler(key_):
    key_fwd, key_bwd, key_kf = jax.random.split(key_, num=3)
    path_y = fwd_ys_sampler(key_fwd, y0)
    vs = path_y[::-1]
    u0_mean, u0_cov = ref_solver(vs[0])
    x0_mean, x0_cov = adhoc_kf(u0_mean, u0_cov, vs)
    return x0_mean + jax.random.normal(key_kf, (d,)) @ jnp.linalg.cholesky(x0_cov)


approx_cond_samples = np.zeros((nsamples, d))
for i in range(nsamples):
    key, subkey = jax.random.split(key)
    approx_cond_sample = conditional_sampler(subkey)
    approx_cond_samples[i] = approx_cond_sample
    print(f'ID: {args.id} | Sample {i}')

# Save results
# np.savez(f'./toy/results/filter-{args.sde}-{args.nparticles}-{args.id}',
#          samples=approx_cond_samples, gp_mean=gp_mean, gp_cov=gp_cov)

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
