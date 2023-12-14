import matplotlib.pyplot as plt
import pytest
import math
import scipy
import jax
import jax.numpy as jnp
import jaxopt
import numpy.testing as npt
from fbs.filters.csmc._csmc2 import csmc
from fbs.filters.csmc.resamplings import killing
from fbs.utils import discretise_lti_sde
from functools import partial

jax.config.update("jax_enable_x64", True)


# LGSSM: test mean, cov, and bivariate cov C(X_t, X_{t+1} | Y_{0:T}) for T = 25


# Non-linear case:
# If x is distributed according to the prior dynamics, then sample y | x, then use CSMC to sample x | y
# This is a Gibbs sampler that keeps the prior distribution of x intact.
# p(x) Gaussian, p(y | x) something weird.

def test_csmc_gibbs():
    pass


def test_csmc_gp_regression():
    """
    x_k = F x_{k-1} + Q_k, Q_k ~ N(0, Q)
    y_k = x_k + R_k, R_k ~ N(0, R)
    """

    def gp_cov(t1, t2):
        return sigma ** 2 * jnp.exp(-jnp.abs(t1[None, :] - t2[:, None]) / ell)

    T = 1
    nsteps = 10
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    nsamples = 1000
    niters = 100

    ell, sigma = 1., 1.

    a, b = -1 / ell, math.sqrt(2 / ell) * sigma
    F, Q = discretise_lti_sde(a * jnp.eye(1), b ** 2 * jnp.eye(1), dt)
    F, Q = jnp.squeeze(F), jnp.squeeze(Q)
    chol_Q = jnp.sqrt(Q)
    R = 1.

    key = jax.random.PRNGKey(666)
    xs = jnp.linalg.cholesky(gp_cov(ts, ts)) @ jax.random.normal(key, (nsteps + 1,))
    ys = xs + math.sqrt(R) * jax.random.normal(key, (nsteps + 1,))

    x0 = xs[0]
    y0 = ys[0]

    def init_sampler(_, nsamples_):
        return x0 * jnp.ones((nsamples_, 1))

    def transition_sampler(us, v_prev, t_prev, key_):
        return us * F + jax.random.normal(key_, us.shape) * chol_Q

    def transition_logpdf(u, u_prev, v_pref, t_prev):
        return jax.scipy.stats.norm.logpdf(u, u_prev * F, chol_Q)

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def measurement_cond_logpdf(v, u_prev, v_prev, t_prev):
        return jax.scipy.stats.norm.logpdf(v, F * u_prev, math.sqrt(R))

    gain = gp_cov(ts, ts) + R * jnp.eye(nsteps + 1)
    chol_gain = jax.scipy.linalg.cho_factor(gain)
    posterior_mean = gp_cov(ts, ts) @ jax.scipy.linalg.cho_solve(chol_gain, ys)
    posterior_cov = gp_cov(ts, ts) - gp_cov(ts, ts) @ jax.scipy.linalg.cho_solve(chol_gain, gp_cov(ts, ts))

    posterior_mean, posterior_cov = posterior_mean[1:], posterior_cov[1:, 1:]

    x_star_0 = jnp.reshape(xs[1:], (-1, 1))
    b_star_0 = jnp.zeros_like(x_star_0)

    x_stars, _ = csmc(key, x_star_0, b_star_0, ys, ts,
                      init_sampler, transition_sampler, transition_logpdf, measurement_cond_logpdf,
                      killing, nsamples, niters, backward=True)
