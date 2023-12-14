import matplotlib.pyplot as plt
import numpy as np
import pytest
import math
import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.filters.csmc.csmc import csmc, csmc_kernel
from fbs.filters.csmc.resamplings import killing
from fbs.utils import discretise_lti_sde
from functools import partial

jax.config.update("jax_enable_x64", True)


# LGSSM: test mean, cov, and bivariate cov C(X_t, X_{t+1} | Y_{0:T}) for T = 25


# Non-linear case:
# If x is distributed according to the prior dynamics, then sample y | x, then use CSMC to sample x | y
# This is a Gibbs sampler that keeps the prior distribution of x intact.
# p(x) Gaussian, p(y | x) something weird.

def gp_cov(t1, t2):
    return sigma ** 2 * jnp.exp(-jnp.abs(t1[None, :] - t2[:, None]) / ell)


def gp_sampler(key_, ts_):
    c = jnp.linalg.cholesky(gp_cov(ts_, ts_))
    return c @ jax.random.normal(key_, ts_.shape)


ell, sigma = 1., 1.
stat_m, stat_var = 0., sigma ** 2

a, b = -1 / ell, math.sqrt(2 / ell) * sigma


@pytest.mark.parametrize('backward', [False, True])
def test_csmc_gp_regression(backward):
    """
    x_k = F x_{k-1} + Q_k, Q_k ~ N(0, Q)
    y_k = x_k + R_k, R_k ~ N(0, R)
    """

    T = 1
    nsteps = 10
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    nsamples = 100
    niters = 1000
    burn_in = 100

    F, Q = discretise_lti_sde(a * jnp.eye(1), b ** 2 * jnp.eye(1), dt)
    F, Q = jnp.squeeze(F), jnp.squeeze(Q)
    chol_Q = jnp.sqrt(Q)
    R = 1.

    key = jax.random.PRNGKey(666)
    xs = jnp.linalg.cholesky(gp_cov(ts, ts)) @ jax.random.normal(key, (nsteps + 1,))

    key, subkey = jax.random.split(key)
    ys = xs + math.sqrt(R) * jax.random.normal(subkey, (nsteps + 1,))

    # Solve GP regression
    cov_ = gp_cov(ts, ts)
    gain = cov_ + R * jnp.eye(nsteps + 1)
    chol_gain = jax.scipy.linalg.cho_factor(gain)
    posterior_mean = cov_ @ jax.scipy.linalg.cho_solve(chol_gain, ys)
    posterior_cov = cov_ - cov_ @ jax.scipy.linalg.cho_solve(chol_gain, cov_)

    def init_sampler(key_, nsamples_):
        return stat_m + math.sqrt(stat_var) * jax.random.normal(key_, (nsamples_,))

    def init_likelihood_logpdf(y0, x0):
        return jax.scipy.stats.norm.logpdf(y0, x0, math.sqrt(R))

    def transition_sampler(x, v_prev, t_prev, key_):
        return x * F + jax.random.normal(key_, x.shape) * chol_Q

    def transition_logpdf(x, x_prev, v_pref, t_prev):
        return jax.scipy.stats.norm.logpdf(x, x_prev * F, chol_Q)

    def likelihood_logpdf(y, x, y_prev, t_prev):
        return jax.scipy.stats.norm.logpdf(y, x, math.sqrt(R))

    x_star_0 = xs
    b_star_0 = jnp.zeros_like(x_star_0, dtype='int64')

    key, subkey = jax.random.split(key)
    x_stars, _ = csmc(subkey,
                      x_star_0, b_star_0,
                      ys, ts,
                      init_sampler, init_likelihood_logpdf,
                      transition_sampler, transition_logpdf,
                      likelihood_logpdf,
                      killing, nsamples, niters, backward=backward)
    x_stars = x_stars[burn_in:]

    npt.assert_allclose(jnp.mean(x_stars, axis=0), posterior_mean, rtol=1e-1)
    npt.assert_allclose(jnp.cov(x_stars, rowvar=False), posterior_cov, atol=3e-2)


def test_csmc_gibbs():
    """Gibbs for p(x_{0:T} | y_{0:T}) and p(y_{0:T} | x_{0:T}) using CSMC.
    """
    T = 1
    nsteps = 100
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    nsamples = 100
    niters = 2000
    burn_in = 1000

    F, Q = discretise_lti_sde(a * jnp.eye(1), b ** 2 * jnp.eye(1), dt)
    F, Q = jnp.squeeze(F), jnp.squeeze(Q)
    chol_Q = jnp.sqrt(Q)
    R = 1.

    def emission(x):
        return jnp.tanh(x)

    @jax.jit
    def sampler_ys_cond_xs(xs_, key_):
        return emission(xs_) + math.sqrt(R) * jax.random.normal(key_, xs_.shape)

    def init_sampler(key_, nsamples_):
        return stat_m + math.sqrt(stat_var) * jax.random.normal(key_, (nsamples_,))

    def init_likelihood_logpdf(y0, x0):
        return jax.scipy.stats.norm.logpdf(y0, emission(x0), math.sqrt(R))

    def transition_sampler(x, v_prev, t_prev, key_):
        return x * F + jax.random.normal(key_, x.shape) * chol_Q

    def transition_logpdf(x, x_prev, v_pref, t_prev):
        return jax.scipy.stats.norm.logpdf(x, x_prev * F, chol_Q)

    def likelihood_logpdf(y, x, y_prev, t_prev):
        return jax.scipy.stats.norm.logpdf(y, emission(x), math.sqrt(R))

    @jax.jit
    def sampler_xs_cond_ys(ys_, key_, xs_star, bs_star):
        xs_star, bs_star = csmc_kernel(key_,
                                       xs_star, bs_star,
                                       ys_, ts,
                                       init_sampler, init_likelihood_logpdf,
                                       transition_sampler, transition_logpdf,
                                       likelihood_logpdf,
                                       killing, nsamples,
                                       backward=False)
        return xs_star, bs_star

    # Gibbs loop
    key = jax.random.PRNGKey(666)
    gibbs_xss = np.zeros((niters, nsteps + 1))

    key, subkey = jax.random.split(key)
    xs = jnp.linalg.cholesky(gp_cov(ts, ts)) @ jax.random.normal(subkey, (nsteps + 1,))
    bs = jnp.zeros_like(xs, dtype='int64')
    for i in range(niters):
        key, subkey = jax.random.split(key)
        ys = sampler_ys_cond_xs(xs, subkey)
        key, subkey = jax.random.split(key)
        xs, bs = sampler_xs_cond_ys(ys, subkey, xs, bs)
        gibbs_xss[i] = xs

    gibbs_xss = gibbs_xss[burn_in:]

    plt.plot(ts, gibbs_xss.T, c='black', alpha=0.1)
    plt.ylim(-3, 3)
    plt.show()

    keys_ = jax.random.split(key, num=niters - burn_in)
    plt.plot(ts, jax.vmap(gp_sampler, in_axes=[0, None])(keys_, ts).T, c='black', alpha=0.1)
    plt.ylim(-3, 3)
    plt.show()

    npt.assert_allclose(jnp.mean(gibbs_xss, axis=0), np.zeros_like(ts), atol=1e-1)
    npt.assert_allclose(jnp.cov(gibbs_xss, rowvar=False), gp_cov(ts, ts), atol=1e-2)

    print(jnp.mean(gibbs_xss, axis=0))
    print(jnp.cov(gibbs_xss, rowvar=False))
    print(gp_cov(ts, ts))