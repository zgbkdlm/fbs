import matplotlib.pyplot as plt
import numpy as np
import pytest
import math
import jax
import jax.numpy as jnp
import numpy.testing as npt
from fbs.samplers.csmc.csmc import csmc_kernel
from fbs.samplers.csmc.resamplings import killing, multinomial
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
def test_csmc_gibbs(backward):
    """Gibbs for p(x_{0:T} | y_{0:T}) and p(y_{0:T} | x_{0:T}) using CSMC.
    """
    T = 10
    nsteps = 10
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    nparticles = 10
    niters = 2000
    burnin = 1000

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
        return 0 + math.sqrt(stat_var) * jax.random.normal(key_, (nsamples_,))

    def init_likelihood_logpdf(y0, x0, _):
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
                                       killing, nparticles,
                                       backward=backward)
        return xs_star, bs_star

    # bs = jnp.zeros((nsteps + 1), dtype=int)

    # @partial(jax.jit, static_argnums=(2,))
    # def few_iters(key_, xs_star_, n_iters):
    #     keys__ = jax.random.split(key_, num=n_iters)
    #
    #     def body_fun(carry, key__):
    #         key__, subkey = jax.random.split(key__)
    #         xs_star, bs_star = carry
    #         ys_ = sampler_ys_cond_xs(xs_star, subkey)
    #         xs_star, bs_star = sampler_xs_cond_ys(ys_, key__, xs_star, bs_star)
    #         return (xs_star, bs_star), None
    #
    #     (xs_star_, _), _ = jax.lax.scan(body_fun, (xs_star_, bs), keys__)
    #     return xs_star_
    #
    # key = jax.random.PRNGKey(666)
    # key_prior, key_csmc = jax.random.split(key, 2)
    # keys_ = jax.random.split(key_prior, num=niters)
    # prior_samples = jax.vmap(gp_sampler, in_axes=[0, None])(keys_, ts)
    #
    # keys_ = jax.random.split(key_csmc, num=niters)
    # gibbs_samples = jax.vmap(few_iters, in_axes=[0, 0, None])(keys_, prior_samples, 2)

    key = jax.random.PRNGKey(666)
    gibbs_samples = np.zeros((niters, nsteps + 1))

    bs_star = jnp.zeros((nsteps + 1), dtype=int)
    key, subkey = jax.random.split(key)
    xs_star = gp_sampler(subkey, ts)
    for i in range(niters):
        key, subkey = jax.random.split(key)
        ys = sampler_ys_cond_xs(xs_star, subkey)
        key, subkey = jax.random.split(key)
        xs_star, bs_star = sampler_xs_cond_ys(ys, subkey, xs_star, bs_star)
        gibbs_samples[i] = xs_star

    gibbs_samples = gibbs_samples[burnin:]
    cov_gibbs = np.cov(gibbs_samples, rowvar=False)
    npt.assert_allclose(jnp.mean(gibbs_samples, axis=0), np.zeros_like(ts), atol=1e-1)
    npt.assert_allclose(np.diag(cov_gibbs), np.diag(gp_cov(ts, ts)), rtol=1e-1, atol=1e-1)
    npt.assert_allclose(cov_gibbs, gp_cov(ts, ts), atol=2e-1)
